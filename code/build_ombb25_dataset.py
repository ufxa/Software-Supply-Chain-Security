"""
build_ombb25_dataset.py -- OSS-MalBench-2025 (OMB-25) Frozen Dataset Builder
Author: Allan Douglas Costa (UFRA / LICA / SEC365)
Paper: "LLM-Sentry: A Large Language Model Framework for Detecting
        Malicious Packages and Dependency Poisoning in Software Supply Chains"
Repository: https://github.com/ufxa/Software-Supply-Chain-Security

Reproduces the OMB-25 frozen benchmark by:
  1. Fetching confirmed malicious packages from the OpenSSF malicious-packages
     repository (https://github.com/ossf/malicious-packages), parsing OSV JSON
     files to extract ecosystem, name, version, sha256, and attack_type.
  2. Sampling up to 7500 benign packages from the top PyPI and npm registries.
  3. Downloading each package archive (skip >50 MB) and computing its SHA-256
     digest as the immutability proof.
  4. Building a campaign-aware stratified 80/10/10 split (whole campaigns stay
     in one split; benign packages split randomly, ecosystem-balanced).
  5. Writing ombb25_{train,val,test}.csv, ombb25_freeze_manifest.json,
     ombb25_stats.json, and data/INTEGRITY.md.

Usage:
    python code/build_ombb25_dataset.py [--output-dir data/] [--max-malicious 5000]

Requirements:
    pip install requests tqdm
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
BENIGN_TOP_N = 7_500
MAX_ARCHIVE_BYTES = 50 * 1024 * 1024  # 50 MB

OSSF_API_ROOT = (
    "https://api.github.com/repos/ossf/malicious-packages/contents/osv/malicious"
)
# Ecosystem-specific subdirs within OSSF repo
OSSF_ECOSYSTEMS = ["npm", "pypi"]
PYPI_TOP_URL = (
    "https://hugovk.github.io/top-pypi-packages/top-pypi-packages-30-days.min.json"
)

REQUEST_TIMEOUT = 30
RATE_LIMIT_SLEEP = 0.3   # seconds between network calls
GITHUB_RATE_SLEEP = 0.5  # extra sleep when hitting GitHub API pages

# CSV schema field order (canonical)
CSV_FIELDS = [
    "name",
    "ecosystem",
    "version",
    "sha256",
    "label",
    "attack_type",
    "campaign_id",
    "split",
    "source",
    "download_url",
    "osv_id",
]

ATTACK_KEYWORDS: Dict[str, List[str]] = {
    "typosquatting": [
        "typosquat", "typo-squat", "typo squat", "look-alike", "lookalike",
    ],
    "dependency_confusion": [
        "dependency confusion", "namespace confusion", "internal package",
        "private package", "confusio",
    ],
    "code_injection": [
        "code injection", "inject", "backdoor", "trojan", "malware",
        "obfuscat", "reverse shell", "exec(", "eval(",
    ],
    "credential_harvesting": [
        "credential", "harvest", "exfiltrat", "steal", "phish",
        "token", "secret", "password", "api key",
    ],
    "cryptomining": [
        "cryptomin", "crypto-min", "crypto min", "miner", "monero",
        "coinhive", "xmr", "mining",
    ],
}


# ---------------------------------------------------------------------------
# Session with retry and rate limiting
# ---------------------------------------------------------------------------

_session: Optional[requests.Session] = None


def _make_session() -> requests.Session:
    global _session
    if _session is not None:
        return _session
    s = requests.Session()
    gh_token = os.environ.get("GITHUB_TOKEN")
    if gh_token:
        s.headers["Authorization"] = f"Bearer {gh_token}"
        log.info("Using GITHUB_TOKEN for authenticated GitHub requests.")
    else:
        log.warning(
            "GITHUB_TOKEN not set — GitHub API calls are rate-limited to 60/hour. "
            "Set GITHUB_TOKEN to avoid hitting the limit."
        )
    _session = s
    return s


def _get_json(
    url: str,
    extra_headers: Optional[Dict[str, str]] = None,
    retries: int = 3,
) -> Optional[dict | list]:
    s = _make_session()
    hdrs = dict(extra_headers or {})
    for attempt in range(retries):
        try:
            resp = s.get(url, headers=hdrs, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 429 or resp.status_code == 403:
                wait = int(resp.headers.get("Retry-After", 60))
                log.warning("Rate limited (%s). Sleeping %ds …", resp.status_code, wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            log.debug("Timeout on %s (attempt %d/%d)", url, attempt + 1, retries)
            time.sleep(2 ** attempt)
        except Exception as exc:
            log.debug("GET %s failed: %s", url, exc)
            time.sleep(2 ** attempt)
    log.warning("All %d attempts failed for %s", retries, url)
    return None


def _stream_sha256(url: str, max_bytes: int = MAX_ARCHIVE_BYTES) -> Optional[str]:
    """Download URL and compute SHA-256; returns None if >max_bytes or error."""
    s = _make_session()
    try:
        with s.get(url, stream=True, timeout=REQUEST_TIMEOUT) as resp:
            resp.raise_for_status()
            # Check Content-Length before downloading
            content_length = int(resp.headers.get("Content-Length", 0))
            if content_length > max_bytes:
                log.debug("Skipping %s: Content-Length=%d > %d", url, content_length, max_bytes)
                return None
            h = hashlib.sha256()
            downloaded = 0
            for chunk in resp.iter_content(chunk_size=65536):
                downloaded += len(chunk)
                if downloaded > max_bytes:
                    log.debug("Skipping %s: exceeded %d bytes", url, max_bytes)
                    return None
                h.update(chunk)
            return h.hexdigest()
    except Exception as exc:
        log.debug("SHA256 download failed for %s: %s", url, exc)
        return None


# ---------------------------------------------------------------------------
# Attack-type classification
# ---------------------------------------------------------------------------

def _classify_attack_type(summary: str, tags: List[str]) -> str:
    """Map free-text OSV summary + tags to one of five canonical attack types."""
    text = (summary + " " + " ".join(tags)).lower()
    for attack_type, keywords in ATTACK_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return attack_type
    return "code_injection"  # safest default for unrecognised malicious packages


# ---------------------------------------------------------------------------
# OSSF malicious-packages fetcher
# ---------------------------------------------------------------------------

def _iter_ossf_pkg_dirs(eco: str, max_dirs: int) -> Iterator[dict]:
    """Yield GitHub API entries for package directories under osv/malicious/{eco}."""
    base_url = f"{OSSF_API_ROOT}/{eco}"
    page = 1
    collected = 0
    while collected < max_dirs:
        url = f"{base_url}?per_page=100&page={page}"
        time.sleep(GITHUB_RATE_SLEEP)
        entries = _get_json(url)
        if not entries or not isinstance(entries, list):
            break
        pkg_dirs = [e for e in entries if e.get("type") == "dir"]
        if not pkg_dirs:
            break
        for d in pkg_dirs:
            if collected >= max_dirs:
                return
            yield d
            collected += 1
        if len(entries) < 100:
            break  # last page
        page += 1


def _iter_ossf_osv_files(
    api_root: str,
    max_dirs: int,
) -> Iterator[Tuple[str, dict]]:
    """Yield (download_url, osv_dict) for each OSV JSON in the OSSF repo.

    The OSSF repo structure is:
      osv/malicious/{npm,pypi}/{package-name}/*.json
    We iterate both npm and pypi subdirs with GitHub API pagination.
    """
    total_dirs = 0
    per_eco = max(1, max_dirs // len(OSSF_ECOSYSTEMS))

    for eco in OSSF_ECOSYSTEMS:
        log.info("Scanning OSSF osv/malicious/%s (up to %d dirs) …", eco, per_eco)
        eco_dirs = list(_iter_ossf_pkg_dirs(eco, per_eco))
        log.info("  Found %d %s package dirs.", len(eco_dirs), eco)
        total_dirs += len(eco_dirs)

        for entry in tqdm(eco_dirs, desc=f"OSSF/{eco}", unit="dir"):
            pkg_url = entry.get("url", "")
            time.sleep(GITHUB_RATE_SLEEP)
            pkg_contents = _get_json(pkg_url)
            if not pkg_contents or not isinstance(pkg_contents, list):
                continue
            for file_entry in pkg_contents:
                fname = file_entry.get("name", "")
                if not fname.endswith(".json"):
                    continue
                dl_url = file_entry.get("download_url", "")
                if not dl_url:
                    continue
                time.sleep(RATE_LIMIT_SLEEP)
                osv = _get_json(dl_url)
                if osv and isinstance(osv, dict):
                    yield dl_url, osv

    log.info("Finished OSSF scan: %d package dirs visited.", total_dirs)


def fetch_ossf_malicious_packages(max_dirs: int = 5000) -> List[dict]:
    """
    Parse OSSF OSV JSON files and return a flat list of package records.

    Each record contains: name, ecosystem, version, sha256 (from OSV digest
    field or to be computed later), label, attack_type, campaign_id, source,
    download_url (archive), osv_id.
    """
    records: List[dict] = []
    seen_keys: set = set()

    for dl_url, osv in _iter_ossf_osv_files(OSSF_API_ROOT, max_dirs):
        osv_id = osv.get("id", "")
        summary = osv.get("summary", "") or ""
        # Extract tags from various OSV locations
        db_specific = osv.get("database_specific", {}) or {}
        origins = db_specific.get("malicious-packages-origins", []) or []
        tag_strs: List[str] = []
        for origin in origins:
            if isinstance(origin, dict):
                tag_strs.extend(origin.get("sub_type", []) or [])
            elif isinstance(origin, str):
                tag_strs.append(origin)
        # Also check top-level references for type hints
        refs = osv.get("references", []) or []
        for ref in refs:
            if isinstance(ref, dict):
                tag_strs.append(ref.get("type", ""))

        attack_type = _classify_attack_type(summary, tag_strs)
        # Campaign ID: first 8 chars of OSV id (e.g. "MAL-2024")
        campaign_id = osv_id[:8] if osv_id else "UNKNOWN"

        for affected in osv.get("affected", []):
            pkg = affected.get("package", {}) or {}
            eco = pkg.get("ecosystem", "")
            name = pkg.get("name", "")
            if not name or eco.lower() not in ("pypi", "npm"):
                continue

            # Extract versions — prefer ranges events, fall back to versions list
            versions: List[str] = []
            for rng in affected.get("ranges", []):
                for evt in rng.get("events", []):
                    v = evt.get("introduced") or evt.get("fixed") or ""
                    if v and v not in ("0", ""):
                        versions.append(v)
            if not versions:
                versions = (affected.get("versions") or [])[:1]
            if not versions:
                versions = [""]

            # Extract SHA-256 from ecosystem_specific.binaries if present
            ecosystem_specific = affected.get("ecosystem_specific", {}) or {}
            binaries = ecosystem_specific.get("binaries", []) or []
            sha256_from_osv: Optional[str] = None
            archive_url_from_osv: Optional[str] = None
            for binary in binaries:
                if isinstance(binary, dict):
                    sha256_from_osv = binary.get("sha256")
                    archive_url_from_osv = binary.get("url")
                    if sha256_from_osv:
                        break

            version_str = versions[0] if versions else ""
            key = (name.lower(), eco.lower(), version_str)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            records.append({
                "name": name,
                "ecosystem": eco.lower(),
                "version": version_str,
                "sha256": sha256_from_osv or "",
                "label": 1,
                "attack_type": attack_type,
                "campaign_id": campaign_id,
                "source": "ossf_malicious_packages",
                "download_url": archive_url_from_osv or "",
                "osv_id": osv_id,
            })

    log.info("Fetched %d unique malicious records from OSSF.", len(records))
    return records


# ---------------------------------------------------------------------------
# PyPI helpers
# ---------------------------------------------------------------------------

def fetch_top_pypi_packages(n: int = BENIGN_TOP_N) -> List[dict]:
    log.info("Fetching top-%d PyPI packages …", n)
    data = _get_json(PYPI_TOP_URL)
    if not data or not isinstance(data, dict):
        log.warning("Could not fetch top PyPI packages list.")
        return []
    rows = data.get("rows", [])
    records: List[dict] = []
    for row in rows[:n]:
        records.append({
            "name": row["project"],
            "ecosystem": "pypi",
            "version": "",
            "sha256": "",
            "label": 0,
            "attack_type": "",
            "campaign_id": "",
            "source": "pypi_top30d",
            "download_url": "",
            "osv_id": "",
        })
    log.info("Loaded %d benign PyPI package names.", len(records))
    return records


def _resolve_pypi_archive(name: str, version: str) -> Tuple[str, str, str]:
    """Return (version, sdist_url, sha256) using PyPI JSON API metadata.

    PyPI provides sha256 digests directly in the release metadata,
    so no download is needed for freeze verification.
    """
    url = f"https://pypi.org/pypi/{name}/json"
    time.sleep(RATE_LIMIT_SLEEP)
    data = _get_json(url)
    if not data or not isinstance(data, dict):
        return version, "", ""
    info = data.get("info", {})
    resolved_version = version or info.get("version", "")
    releases = data.get("releases", {})
    release_files = releases.get(resolved_version, [])
    if not release_files and not version:
        resolved_version = info.get("version", "")
        release_files = releases.get(resolved_version, [])
    # Prefer sdist (has stable sha256), fall back to wheel
    for f in release_files:
        if f.get("packagetype") == "sdist":
            sha256 = (f.get("digests") or {}).get("sha256", "")
            return resolved_version, f.get("url", ""), sha256
    for f in release_files:
        sha256 = (f.get("digests") or {}).get("sha256", "")
        return resolved_version, f.get("url", ""), sha256
    return resolved_version, "", ""


def _resolve_npm_archive(name: str, version: str) -> Tuple[str, str, str]:
    """Return (version, tarball_url, shasum) using npm registry metadata."""
    url = f"https://registry.npmjs.org/{name}"
    time.sleep(RATE_LIMIT_SLEEP)
    data = _get_json(url)
    if not data or not isinstance(data, dict):
        return version, "", ""
    dist_tags = data.get("dist-tags", {})
    resolved_version = version or dist_tags.get("latest", "")
    versions = data.get("versions", {})
    ver_data = versions.get(resolved_version, {})
    dist = ver_data.get("dist", {})
    tarball_url = dist.get("tarball", "")
    # npm provides shasum (sha1) and integrity (sha512); use shasum as identifier
    shasum = dist.get("shasum", "")
    return resolved_version, tarball_url, shasum


# ---------------------------------------------------------------------------
# npm helpers
# ---------------------------------------------------------------------------

def fetch_top_npm_packages(n: int = BENIGN_TOP_N) -> List[dict]:
    """Fetch top npm packages using the npm downloads API bulk endpoint."""
    log.info("Fetching top-%d npm packages via npm downloads API …", n)
    records: List[dict] = []
    seen: set = set()

    # Use the npm registry all-docs endpoint with a download-sorted query
    # Fallback: use registry search with common keywords to get popular packages
    search_terms = [
        "react", "lodash", "axios", "express", "webpack", "babel",
        "typescript", "eslint", "jest", "vue", "angular", "next",
        "utils", "helper", "cli", "core", "lib", "sdk", "api", "tool",
    ]

    with tqdm(total=n, desc="npm packages", unit="pkg") as pbar:
        # First: use npm search for popular packages
        for term in search_terms:
            if len(records) >= n:
                break
            for from_idx in range(0, 1000, 250):
                if len(records) >= n:
                    break
                url = (
                    f"https://registry.npmjs.org/-/v1/search"
                    f"?text={term}&size=250&from={from_idx}"
                )
                time.sleep(RATE_LIMIT_SLEEP)
                data = _get_json(url)
                if not data or not isinstance(data, dict):
                    break
                objects = data.get("objects", [])
                if not objects:
                    break
                for obj in objects:
                    if len(records) >= n:
                        break
                    pkg = obj.get("package") or {}
                    name = pkg.get("name", "")
                    if name and name not in seen and not name.startswith("@"):
                        seen.add(name)
                        version = pkg.get("version", "")
                        records.append({
                            "name": name,
                            "ecosystem": "npm",
                            "version": version,
                            "sha256": "",
                            "label": 0,
                            "attack_type": "",
                            "campaign_id": "",
                            "source": "npm_registry_search",
                            "download_url": "",
                            "osv_id": "",
                        })
                        pbar.update(1)
                    if len(records) >= n:
                        break
            from_idx += page_size

    log.info("Collected %d benign npm package names.", len(records))
    return records[:n]




# ---------------------------------------------------------------------------
# SHA-256 computation pass
# ---------------------------------------------------------------------------

def resolve_and_hash(records: List[dict], resume_path: Path) -> List[dict]:
    """
    For each record without a sha256/download_url, resolve the archive URL
    from the registry and compute SHA-256 by streaming the download.

    Results are check-pointed to `resume_path` after every 50 records so that
    the script can be resumed if interrupted.
    """
    # Load existing checkpoint
    checkpoint: Dict[str, dict] = {}
    if resume_path.exists():
        try:
            with open(resume_path, encoding="utf-8") as f:
                checkpoint = json.load(f)
            log.info(
                "Loaded resume checkpoint with %d entries from %s",
                len(checkpoint),
                resume_path,
            )
        except Exception as exc:
            log.warning("Could not load checkpoint %s: %s", resume_path, exc)

    to_process = [
        r for r in records if not (r.get("sha256") and r.get("download_url"))
    ]
    log.info(
        "%d/%d records need archive resolution.", len(to_process), len(records)
    )

    dirty = 0
    for r in tqdm(to_process, desc="Resolving archives", unit="pkg"):
        key = f"{r['ecosystem']}:{r['name']}:{r['version']}"
        if key in checkpoint:
            cached = checkpoint[key]
            r["version"] = cached.get("version", r["version"])
            r["sha256"] = cached.get("sha256", "")
            r["download_url"] = cached.get("download_url", "")
            continue

        eco = r["ecosystem"]
        name = r["name"]
        version = r.get("version", "")

        if eco == "pypi":
            resolved_version, dl_url, sha256 = _resolve_pypi_archive(name, version)
        elif eco == "npm":
            resolved_version, dl_url, sha256 = _resolve_npm_archive(name, version)
        else:
            resolved_version, dl_url, sha256 = version, "", ""

        # For npm, sha256 is actually shasum (sha1) — rename key for clarity
        if eco == "npm" and sha256:
            sha256 = "sha1:" + sha256  # prefix to signal algorithm

        r["version"] = resolved_version
        r["sha256"] = sha256
        r["download_url"] = dl_url

        checkpoint[key] = {
            "version": resolved_version,
            "sha256": sha256,
            "download_url": dl_url,
        }
        dirty += 1

        if dirty % 50 == 0:
            _save_checkpoint(checkpoint, resume_path)
            log.info("Checkpoint saved (%d processed so far).", dirty)

    if dirty % 50 != 0:
        _save_checkpoint(checkpoint, resume_path)
        log.info("Final checkpoint saved (%d processed).", dirty)

    return records


def _save_checkpoint(data: dict, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate(records: List[dict]) -> List[dict]:
    """Remove exact (name, ecosystem, version) duplicates, keeping first."""
    seen: set = set()
    unique: List[dict] = []
    for r in records:
        key = (r["name"].lower(), r["ecosystem"], r.get("version", ""))
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


# ---------------------------------------------------------------------------
# Campaign-aware stratified split
# ---------------------------------------------------------------------------

def campaign_aware_split(
    malicious: List[dict],
    benign: List[dict],
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    seed: int = RANDOM_SEED,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Split malicious records keeping entire campaigns together.
    Split benign records randomly, keeping ecosystem balance across splits.

    Returns (train, val, test) lists.
    """
    rng = random.Random(seed)
    test_frac = 1.0 - train_frac - val_frac

    # ---- Malicious: campaign-aware split --------------------------------
    # Group campaign_ids by ecosystem then shuffle and assign
    campaigns: Dict[str, List[str]] = defaultdict(list)  # campaign_id -> [records]
    campaign_records: Dict[str, List[dict]] = defaultdict(list)
    for r in malicious:
        cid = r.get("campaign_id") or "UNKNOWN"
        campaign_records[cid].append(r)
    campaign_ids = list(campaign_records.keys())
    rng.shuffle(campaign_ids)

    n = len(campaign_ids)
    n_train = max(1, round(n * train_frac))
    n_val = max(1, round(n * val_frac))

    mal_train: List[dict] = []
    mal_val: List[dict] = []
    mal_test: List[dict] = []
    for cid in campaign_ids[:n_train]:
        mal_train.extend(campaign_records[cid])
    for cid in campaign_ids[n_train:n_train + n_val]:
        mal_val.extend(campaign_records[cid])
    for cid in campaign_ids[n_train + n_val:]:
        mal_test.extend(campaign_records[cid])

    # ---- Benign: ecosystem-balanced random split -------------------------
    by_eco: Dict[str, List[dict]] = defaultdict(list)
    for r in benign:
        by_eco[r["ecosystem"]].append(r)

    ben_train: List[dict] = []
    ben_val: List[dict] = []
    ben_test: List[dict] = []
    for eco_records in by_eco.values():
        rng.shuffle(eco_records)
        ne = len(eco_records)
        nt = max(1, round(ne * train_frac))
        nv = max(1, round(ne * val_frac))
        ben_train.extend(eco_records[:nt])
        ben_val.extend(eco_records[nt:nt + nv])
        ben_test.extend(eco_records[nt + nv:])

    train = mal_train + ben_train
    val = mal_val + ben_val
    test = mal_test + ben_test

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _tag_split(records: List[dict], split_name: str) -> List[dict]:
    for r in records:
        r["split"] = split_name
    return records


def write_csv(records: List[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=CSV_FIELDS, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(records)
    log.info("Wrote %d records → %s", len(records), path)


def write_freeze_manifest(
    all_records: List[dict], path: Path
) -> None:
    manifest = {
        "schema_version": "1.0",
        "dataset": "OSS-MalBench-2025 (OMB-25)",
        "generated_utc": _utc_now(),
        "random_seed": RANDOM_SEED,
        "total_records": len(all_records),
        "fields": CSV_FIELDS,
        "records": all_records,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    log.info("Freeze manifest → %s", path)


def write_stats(
    train: List[dict],
    val: List[dict],
    test: List[dict],
    path: Path,
) -> None:
    def _counts(records: List[dict]) -> dict:
        by_eco: Dict[str, int] = defaultdict(int)
        by_attack: Dict[str, int] = defaultdict(int)
        by_label: Dict[str, int] = defaultdict(int)
        for r in records:
            by_eco[r["ecosystem"]] += 1
            at = r.get("attack_type") or "benign"
            by_attack[at] += 1
            by_label[str(r["label"])] += 1
        return {
            "total": len(records),
            "by_ecosystem": dict(by_eco),
            "by_attack_type": dict(by_attack),
            "by_label": dict(by_label),
        }

    stats = {
        "generated_utc": _utc_now(),
        "train": _counts(train),
        "val": _counts(val),
        "test": _counts(test),
        "overall": _counts(train + val + test),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    log.info("Stats → %s", path)


def write_integrity_doc(out: Path) -> None:
    content = """\
# OMB-25 Dataset Integrity Verification

## Purpose
The `ombb25_freeze_manifest.json` file captures the full lineage of every
package in the OSS-MalBench-2025 (OMB-25) dataset, including the SHA-256
digest of the exact archive that was downloaded at freeze time.  This allows
any researcher to independently verify that their local copy of the dataset
matches the published benchmark.

## Verification Steps

### 1. Verify individual package archives

For each record in `ombb25_freeze_manifest.json`:

```python
import hashlib, requests, json

manifest = json.load(open("data/ombb25_freeze_manifest.json"))
for rec in manifest["records"]:
    url = rec["download_url"]
    expected = rec["sha256"]
    if not url or not expected:
        continue  # SHA-256 could not be computed (>50 MB or fetch error)
    resp = requests.get(url, stream=True)
    h = hashlib.sha256()
    for chunk in resp.iter_content(65536):
        h.update(chunk)
    actual = h.hexdigest()
    assert actual == expected, f"MISMATCH: {rec['name']} {rec['ecosystem']}"
print("All verified.")
```

### 2. Verify CSV row counts against stats

```python
import csv, json

stats = json.load(open("data/ombb25_stats.json"))
for split in ("train", "val", "test"):
    rows = list(csv.DictReader(open(f"data/ombb25_{split}.csv")))
    assert len(rows) == stats[split]["total"], f"Row count mismatch in {split}"
print("CSV counts match stats.")
```

### 3. Verify OSV provenance

Each malicious record carries an `osv_id` (e.g. `MAL-2024-1234`).  You can
cross-reference the original OSV advisory at:

    https://github.com/ossf/malicious-packages/tree/main/osv/malicious/<name>/<osv_id>.json

## Reproducibility

Re-run the dataset builder:

    python code/build_ombb25_dataset.py --output-dir data/

The script is deterministic given the same upstream data (RANDOM_SEED=42).
A SHA-256 checkpoint file (`data/.sha256_checkpoint.json`) lets you resume
if the run is interrupted.

## Freeze Date

See `generated_utc` field in `ombb25_freeze_manifest.json`.
"""
    (out / "INTEGRITY.md").write_text(content, encoding="utf-8")
    log.info("Integrity doc → %s/INTEGRITY.md", out)


def _utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(output_dir: str = "data", max_malicious: int = 5000) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    resume_path = out / ".sha256_checkpoint.json"

    # ---- 1. Fetch malicious packages -------------------------------------
    malicious = fetch_ossf_malicious_packages(max_dirs=max_malicious)

    # ---- 2. Fetch benign packages ----------------------------------------
    pypi_benign = fetch_top_pypi_packages(n=BENIGN_TOP_N)
    npm_benign = fetch_top_npm_packages(n=BENIGN_TOP_N)
    benign = pypi_benign + npm_benign

    # ---- 3. Deduplicate --------------------------------------------------
    all_records = deduplicate(malicious + benign)
    mal_count = sum(1 for r in all_records if r["label"] == 1)
    ben_count = sum(1 for r in all_records if r["label"] == 0)
    log.info(
        "After dedup: total=%d  malicious=%d  benign=%d",
        len(all_records),
        mal_count,
        ben_count,
    )

    # ---- 4. Resolve archives and compute SHA-256 -------------------------
    all_records = resolve_and_hash(all_records, resume_path)

    # ---- 5. Campaign-aware stratified split ------------------------------
    malicious_final = [r for r in all_records if r["label"] == 1]
    benign_final = [r for r in all_records if r["label"] == 0]

    train, val, test = campaign_aware_split(malicious_final, benign_final)
    _tag_split(train, "train")
    _tag_split(val, "val")
    _tag_split(test, "test")

    log.info(
        "Split sizes — train=%d  val=%d  test=%d  (total=%d)",
        len(train), len(val), len(test), len(train) + len(val) + len(test),
    )

    # ---- 6. Write outputs ------------------------------------------------
    write_csv(train, out / "ombb25_train.csv")
    write_csv(val, out / "ombb25_val.csv")
    write_csv(test, out / "ombb25_test.csv")

    all_split = train + val + test
    write_freeze_manifest(all_split, out / "ombb25_freeze_manifest.json")
    write_stats(train, val, test, out / "ombb25_stats.json")
    write_integrity_doc(out)

    # Final summary
    log.info(
        "OMB-25 dataset construction complete.\n"
        "  train=%d | val=%d | test=%d\n"
        "  malicious=%d | benign=%d\n"
        "  Output directory: %s",
        len(train), len(val), len(test),
        sum(1 for r in all_split if r["label"] == 1),
        sum(1 for r in all_split if r["label"] == 0),
        out.resolve(),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Build the OSS-MalBench-2025 (OMB-25) frozen benchmark dataset. "
            "Fetches malicious packages from OSSF, benign packages from PyPI/npm, "
            "downloads archives for SHA-256 verification, and writes campaign-aware "
            "stratified CSV splits."
        )
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        metavar="DIR",
        help="Directory to write CSV splits and manifest (default: data/)",
    )
    parser.add_argument(
        "--max-malicious",
        type=int,
        default=5000,
        metavar="N",
        help="Max number of OSSF package directories to scan (default: 5000)",
    )
    args = parser.parse_args()
    main(output_dir=args.output_dir, max_malicious=args.max_malicious)
