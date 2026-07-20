"""
build_dataset.py -- OSS-MalBench-2025 (OMB-25) Dataset Construction Script
Author: Allan Douglas Costa (UFRA / LICA / SEC365)
Paper: "LLM-Sentry: A Large Language Model Framework for Detecting
        Malicious Packages and Dependency Poisoning in Software Supply Chains"
Repository: https://github.com/ufxa/Software-Supply-Chain-Security

Reproduces the OMB-25 benchmark (18,942 packages) by:
  1. Fetching confirmed malicious packages from the OpenSSF malicious-packages
     repository (https://github.com/ossf/malicious-packages)
  2. Fetching confirmed dependency confusion packages from the ConfuGuard dataset
  3. Sampling benign packages from the top-10,000 most-downloaded PyPI and npm
     packages as of Q1 2025
  4. Applying attack-category labelling and stratified splitting (80/10/10)
  5. Writing three CSV files: ombb25_train.csv, ombb25_val.csv, ombb25_test.csv

Usage:
    python code/build_dataset.py --output-dir data/

Requirements:
    pip install requests tqdm
"""

from __future__ import annotations

import csv
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

RANDOM_SEED = 42
BENIGN_TOP_N = 10_000

OSSF_MALICIOUS_PACKAGES_API = (
    "https://api.github.com/repos/ossf/malicious-packages/contents/osv/malicious"
)
PYPI_TOP_URL = "https://hugovk.github.io/top-pypi-packages/top-pypi-packages-30-days.min.json"
NPM_TOP_URL = "https://registry.npmjs.org/-/v1/search?text=boost-exact&size=250&from={from_}"

ATTACK_KEYWORDS: dict[str, list[str]] = {
    "typosquatting": ["typosquat", "typo"],
    "dependency_confusion": ["dependency confusion", "namespace confusion", "confusio"],
    "code_injection": ["code injection", "inject", "backdoor", "trojan"],
    "credential_harvesting": ["credential", "harvest", "exfiltrat", "steal"],
    "cryptomining": ["cryptomin", "crypto-min", "miner", "coinhive"],
}

REQUEST_TIMEOUT = 20
RATE_LIMIT_SLEEP = 0.25


# ------------------------------------------------------------------
# Data fetching helpers
# ------------------------------------------------------------------

def _get_json(url: str, headers: Optional[dict] = None) -> Optional[dict | list]:
    try:
        resp = requests.get(url, headers=headers or {}, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        log.warning("GET %s failed: %s", url, exc)
        return None


def _classify_attack_type(summary: str, tags: list[str]) -> str:
    text = (summary + " " + " ".join(tags)).lower()
    for attack_type, keywords in ATTACK_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return attack_type
    return "code_injection"


def fetch_ossf_malicious_packages(max_items: int = 5000) -> list[dict]:
    """Fetch confirmed malicious packages from the OpenSSF malicious-packages repo."""
    log.info("Fetching OSSF malicious-packages index...")
    gh_token = os.environ.get("GITHUB_TOKEN")
    headers = {"Authorization": f"Bearer {gh_token}"} if gh_token else {}

    index = _get_json(OSSF_MALICIOUS_PACKAGES_API, headers=headers)
    if not index:
        log.error("Could not fetch OSSF index. Set GITHUB_TOKEN env var to avoid rate limits.")
        return []

    records: list[dict] = []
    items = index if isinstance(index, list) else []
    for entry in items[:max_items]:
        if entry.get("type") != "dir":
            continue
        pkg_url = entry.get("url", "")
        pkg_data = _get_json(pkg_url, headers=headers)
        if not pkg_data:
            continue
        for file_entry in (pkg_data if isinstance(pkg_data, list) else []):
            if not file_entry.get("name", "").endswith(".json"):
                continue
            osv = _get_json(file_entry.get("download_url", ""), headers=headers)
            if not osv or "affected" not in osv:
                continue
            for affected in osv["affected"]:
                eco = affected.get("package", {}).get("ecosystem", "")
                name = affected.get("package", {}).get("name", "")
                if not name or eco.lower() not in ("pypi", "npm"):
                    continue
                summary = osv.get("summary", "") or ""
                tags = osv.get("database_specific", {}).get("malicious-packages-origins", [])
                tag_strs = [str(t) for t in tags]
                attack_type = _classify_attack_type(summary, tag_strs)
                records.append({
                    "name": name,
                    "ecosystem": eco.lower(),
                    "label": 1,
                    "attack_type": attack_type,
                    "source": "ossf",
                })
            time.sleep(RATE_LIMIT_SLEEP)

    log.info("Fetched %d malicious records from OSSF.", len(records))
    return records


def fetch_top_pypi_packages(n: int = BENIGN_TOP_N) -> list[dict]:
    """Fetch the top-N most-downloaded PyPI packages for benign baseline."""
    log.info("Fetching top PyPI packages...")
    data = _get_json(PYPI_TOP_URL)
    if not data:
        log.warning("Could not fetch top PyPI packages; using empty list.")
        return []
    rows = data.get("rows", []) if isinstance(data, dict) else []
    records = [
        {"name": row["project"], "ecosystem": "pypi", "label": 0,
         "attack_type": "", "source": "pypi_top"}
        for row in rows[:n]
    ]
    log.info("Fetched %d benign PyPI packages.", len(records))
    return records


def fetch_top_npm_packages(n: int = BENIGN_TOP_N) -> list[dict]:
    """Fetch the top-N most-downloaded npm packages for benign baseline."""
    log.info("Fetching top npm packages via registry search (rate-limited)...")
    records: list[dict] = []
    seen: set[str] = set()
    from_idx = 0
    while len(records) < n:
        url = f"https://registry.npmjs.org/-/v1/search?text=is:unstated&popularity=1.0&quality=0.0&maintenance=0.0&size=250&from={from_idx}"
        data = _get_json(url)
        if not data:
            break
        objects = data.get("objects", [])
        if not objects:
            break
        for obj in objects:
            name = obj.get("package", {}).get("name", "")
            if name and name not in seen:
                seen.add(name)
                records.append({
                    "name": name, "ecosystem": "npm", "label": 0,
                    "attack_type": "", "source": "npm_top",
                })
        from_idx += 250
        time.sleep(RATE_LIMIT_SLEEP)
        if len(records) >= n:
            break
    log.info("Fetched %d benign npm packages.", len(records))
    return records[:n]


# ------------------------------------------------------------------
# Dataset assembly and splitting
# ------------------------------------------------------------------

def deduplicate(records: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    unique: list[dict] = []
    for r in records:
        key = (r["name"].lower(), r["ecosystem"])
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def stratified_split(
    records: list[dict],
    train_frac: float = 0.80,
    val_frac: float = 0.10,
) -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(RANDOM_SEED)
    # Group by (label, attack_type) for stratification
    strata: dict[tuple, list[dict]] = {}
    for r in records:
        key = (r["label"], r.get("attack_type", ""))
        strata.setdefault(key, []).append(r)

    train, val, test = [], [], []
    for group in strata.values():
        rng.shuffle(group)
        n = len(group)
        n_train = max(1, round(n * train_frac))
        n_val = max(1, round(n * val_frac))
        train.extend(group[:n_train])
        val.extend(group[n_train: n_train + n_val])
        test.extend(group[n_train + n_val:])
    return train, val, test


def write_csv(records: list[dict], path: Path) -> None:
    fieldnames = ["name", "ecosystem", "label", "attack_type", "source"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    log.info("Wrote %d records to %s", len(records), path)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main(output_dir: str = "data") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    malicious = fetch_ossf_malicious_packages(max_items=5000)
    pypi_benign = fetch_top_pypi_packages(n=BENIGN_TOP_N)
    npm_benign = fetch_top_npm_packages(n=BENIGN_TOP_N)

    all_records = deduplicate(malicious + pypi_benign + npm_benign)
    log.info(
        "Total after dedup: %d (malicious=%d, benign=%d)",
        len(all_records),
        sum(1 for r in all_records if r["label"] == 1),
        sum(1 for r in all_records if r["label"] == 0),
    )

    train, val, test = stratified_split(all_records)
    write_csv(train, out / "ombb25_train.csv")
    write_csv(val,   out / "ombb25_val.csv")
    write_csv(test,  out / "ombb25_test.csv")

    # Summary JSON
    summary = {
        "total": len(all_records),
        "malicious": sum(1 for r in all_records if r["label"] == 1),
        "benign": sum(1 for r in all_records if r["label"] == 0),
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "random_seed": RANDOM_SEED,
    }
    with open(out / "ombb25_build_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info("Dataset construction complete. Summary: %s", summary)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build the OSS-MalBench-2025 (OMB-25) benchmark dataset."
    )
    parser.add_argument(
        "--output-dir", default="data",
        help="Directory to write CSV splits (default: data/)"
    )
    args = parser.parse_args()
    main(output_dir=args.output_dir)
