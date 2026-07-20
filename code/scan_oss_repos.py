"""
scan_oss_repos.py -- OSS Repository Scanner for LLM-Sentry Evaluation
Author: Allan Douglas Costa (UFRA / LICA / SEC365)
Paper: "LLM-Sentry: A Large Language Model Framework for Detecting
        Malicious Packages and Dependency Poisoning in Software Supply Chains"
Repository: https://github.com/ufxa/Software-Supply-Chain-Security

This script downloads package archives from npm and PyPI registries,
runs LLM-Sentry analysis, and saves results to a CSV file for evaluation.
NOTE: LLM analysis is skipped for performance; only metadata and behavioral
      static analysis are run during bulk scanning.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

PYPI_JSON_URL = "https://pypi.org/pypi/{package}/json"
NPM_JSON_URL = "https://registry.npmjs.org/{package}"
DOWNLOAD_TIMEOUT = 30
MAX_ARCHIVE_SIZE_MB = 50
SCAN_DELAY_SECONDS = 0.5


# ------------------------------------------------------------------
# Registry clients
# ------------------------------------------------------------------

class PyPIClient:
    """Fetch package metadata and archives from PyPI."""

    def get_metadata(self, package_name: str) -> Optional[dict]:
        url = PYPI_JSON_URL.format(package=package_name)
        try:
            resp = requests.get(url, timeout=DOWNLOAD_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            log.warning("PyPI metadata fetch failed for %s: %s", package_name, exc)
            return None

    def download_latest_sdist(self, package_name: str,
                              dest_dir: str) -> Optional[str]:
        meta = self.get_metadata(package_name)
        if not meta:
            return None
        releases = meta.get("releases", {})
        latest_version = meta["info"]["version"]
        files = releases.get(latest_version, [])
        for f in files:
            if f["filename"].endswith(".tar.gz"):
                size_mb = f.get("size", 0) / (1024 ** 2)
                if size_mb > MAX_ARCHIVE_SIZE_MB:
                    log.warning("Skipping %s: size %.1f MB", f["filename"], size_mb)
                    continue
                dest = Path(dest_dir) / f["filename"]
                r = requests.get(f["url"], timeout=DOWNLOAD_TIMEOUT, stream=True)
                with open(dest, "wb") as fh:
                    for chunk in r.iter_content(8192):
                        fh.write(chunk)
                return str(dest)
        return None


class NpmClient:
    """Fetch package metadata and archives from npm registry."""

    def get_metadata(self, package_name: str) -> Optional[dict]:
        url = NPM_JSON_URL.format(package=package_name)
        try:
            resp = requests.get(url, timeout=DOWNLOAD_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            log.warning("npm metadata fetch failed for %s: %s", package_name, exc)
            return None

    def download_latest_tarball(self, package_name: str,
                                dest_dir: str) -> Optional[str]:
        meta = self.get_metadata(package_name)
        if not meta:
            return None
        dist_tags = meta.get("dist-tags", {})
        latest = dist_tags.get("latest")
        if not latest:
            return None
        version_data = meta.get("versions", {}).get(latest, {})
        tarball_url = version_data.get("dist", {}).get("tarball")
        if not tarball_url:
            return None
        filename = f"{package_name.replace('/', '_')}-{latest}.tgz"
        dest = Path(dest_dir) / filename
        r = requests.get(tarball_url, timeout=DOWNLOAD_TIMEOUT, stream=True)
        size = 0
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(8192):
                fh.write(chunk)
                size += len(chunk)
                if size > MAX_ARCHIVE_SIZE_MB * 1024 ** 2:
                    log.warning("Archive too large, aborting: %s", package_name)
                    return None
        return str(dest)


# ------------------------------------------------------------------
# Main scanning logic
# ------------------------------------------------------------------

def scan_package_list(
    package_list: list[dict],
    output_csv: str,
    skip_llm: bool = True,
) -> None:
    """
    Scan a list of packages and write results to CSV.

    Args:
        package_list: List of dicts with keys: name, ecosystem, label (optional)
        output_csv:   Output CSV file path
        skip_llm:     If True, skip LLM analysis (metadata+behavioral only)
    """
    from llm_sentry import LLMSentry, PackageIngestionModule, MetadataExtractor
    from llm_sentry import BehavioralAnalyzer, PRCSEngine, PackageFeatures
    import numpy as np

    pypi_client = PyPIClient()
    npm_client = NpmClient()

    ingestion = PackageIngestionModule()
    meta_extractor = MetadataExtractor()
    behavioral = BehavioralAnalyzer()
    prcs_engine = PRCSEngine()

    fieldnames = [
        "name", "ecosystem", "version", "true_label",
        "prcs", "pred_label", "s_meta", "s_sem", "s_beh",
        "attack_type", "error",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for pkg_info in package_list:
            name = pkg_info["name"]
            ecosystem = pkg_info.get("ecosystem", "pypi")
            true_label = pkg_info.get("label", -1)
            row = {
                "name": name, "ecosystem": ecosystem,
                "true_label": true_label, "error": "",
            }

            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    if ecosystem == "pypi":
                        archive = pypi_client.download_latest_sdist(name, tmpdir)
                    else:
                        archive = npm_client.download_latest_tarball(name, tmpdir)

                    if not archive:
                        row["error"] = "download_failed"
                        writer.writerow(row)
                        continue

                    vfs = ingestion.ingest(archive)
                    manifest = ingestion.extract_manifest(vfs)
                    install_scripts = ingestion.extract_install_scripts(vfs)
                    meta_vec = meta_extractor.extract(manifest, vfs, install_scripts)
                    _, b_dev = behavioral.analyze_static(install_scripts)

                    features = PackageFeatures(
                        name=name,
                        version=manifest.get("version", "0.0.0"),
                        ecosystem=ecosystem,
                    )
                    vals = meta_vec.tolist()
                    (features.name_dist, features.maint_age, features.version_count,
                     features.dl_velocity, features.script_entropy, features.dep_count,
                     features.repo_present, features.license_present,
                     features.has_postinstall, features.file_count,
                     features.obfusc_ratio, features.url_in_code, features.eval_count,
                     features.b64_count, features.env_access, features.net_call_count,
                     features.fs_write_count, features.pkg_size, features.desc_length,
                     features.keyword_count, features.author_email_entropy,
                     features.publish_date_days) = vals

                    features.code_embedding = None
                    features.llm_confidence = 0.0
                    features.behavioral_deviation = b_dev

                    result = prcs_engine.compute(features)

                    row.update({
                        "version": features.version,
                        "prcs": round(result.prcs, 4),
                        "pred_label": result.label,
                        "s_meta": round(result.s_meta, 4),
                        "s_sem": round(result.s_sem, 4),
                        "s_beh": round(result.s_beh, 4),
                        "attack_type": result.attack_type or "",
                    })

            except Exception as exc:
                log.error("Error scanning %s: %s", name, exc)
                row["error"] = str(exc)[:200]

            writer.writerow(row)
            log.info("Scanned %s [%s] PRCS=%.3f pred=%s true=%s",
                     name, ecosystem,
                     row.get("prcs", 0.0),
                     row.get("pred_label", "?"),
                     true_label)
            time.sleep(SCAN_DELAY_SECONDS)


# ------------------------------------------------------------------
# Example usage
# ------------------------------------------------------------------

EXAMPLE_PACKAGES = [
    # Benign
    {"name": "requests", "ecosystem": "pypi", "label": 0},
    {"name": "numpy", "ecosystem": "pypi", "label": 0},
    {"name": "flask", "ecosystem": "pypi", "label": 0},
    {"name": "express", "ecosystem": "npm", "label": 0},
    {"name": "lodash", "ecosystem": "npm", "label": 0},
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM-Sentry OSS Repository Scanner")
    parser.add_argument("--input-json", default=None,
                        help="JSON file with list of {name, ecosystem, label} dicts")
    parser.add_argument("--output-csv", default="scan_results.csv",
                        help="Output CSV file (default: scan_results.csv)")
    parser.add_argument("--skip-llm", action="store_true", default=True,
                        help="Skip LLM analysis for bulk scanning")
    args = parser.parse_args()

    if args.input_json:
        with open(args.input_json, "r") as f:
            packages = json.load(f)
    else:
        log.info("No input file provided. Using example package list.")
        packages = EXAMPLE_PACKAGES

    scan_package_list(packages, args.output_csv, skip_llm=args.skip_llm)
    log.info("Scan complete. Results written to: %s", args.output_csv)
