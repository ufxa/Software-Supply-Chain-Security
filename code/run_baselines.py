"""
run_baselines.py — Baseline detector implementations for LLM-Sentry comparison.

Implements four baselines on data/ombb25_test.csv:
  1. GuardDog-style rule-based detector
  2. MalOSS-style metadata + logistic regression
  3. CrossLang-style cross-language pattern matching + metadata classifier
  4. Metadata-only baseline (22 features, logistic regression)

Outputs per baseline:
  results/baselines/<name>_predictions.csv
  results/baselines/<name>_metrics.json   (with 5000-iter bootstrap CI)

Usage:
    python code/run_baselines.py

Author: Allan Douglas Costa (UFRA / LICA / SEC365)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np

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
DATA_DIR  = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
BASELINES_DIR = RESULTS_DIR / "baselines"

TRAIN_CSV = DATA_DIR / "ombb25_train.csv"
TEST_CSV  = DATA_DIR / "ombb25_test.csv"

# ---------------------------------------------------------------------------
# Synthetic dataset generation (when real CSVs are absent)
# ---------------------------------------------------------------------------

_ATTACK_TYPES = [
    "credential_harvesting",
    "code_injection",
    "typosquatting",
    "dependency_confusion",
    "cryptomining",
]

_POPULAR_PKGS = [
    "requests", "numpy", "pandas", "flask", "django", "scipy",
    "matplotlib", "tensorflow", "torch", "sklearn", "express",
    "lodash", "react", "axios", "moment", "webpack",
]


def _typosquat_name(rng: random.Random, base: str) -> str:
    ops = ["swap", "duplicate", "replace"]
    op  = rng.choice(ops)
    i   = rng.randint(0, max(0, len(base) - 1))
    if op == "swap" and len(base) > 1:
        lst = list(base)
        lst[i], lst[i - 1] = lst[i - 1], lst[i]
        return "".join(lst)
    if op == "duplicate":
        return base[:i] + base[i] + base[i:]
    chars = "abcdefghijklmnopqrstuvwxyz"
    return base[:i] + rng.choice(chars) + base[i + 1:]


def _generate_row(
    rng: random.Random,
    label: int,
    attack_type: Optional[str],
    idx: int,
    split: str,
) -> dict:
    """Produce one synthetic CSV row with realistic feature values."""
    ecosystem = rng.choice(["pypi", "npm"])

    if label == 1:
        base = rng.choice(_POPULAR_PKGS)
        if attack_type == "typosquatting":
            name = _typosquat_name(rng, base)
        elif attack_type == "dependency_confusion":
            name = f"internal-{base}-{rng.randint(1, 99)}"
        else:
            name = f"{base}-{rng.randint(100, 999)}"
    else:
        name = f"legit-pkg-{idx}"

    name_dist = (
        rng.randint(1, 3) if (label == 1 and attack_type == "typosquatting")
        else rng.randint(5, 20)
    )
    maint_age        = rng.uniform(0, 30) if label == 1 else rng.uniform(180, 3000)
    version_count    = rng.randint(1, 3) if label == 1 else rng.randint(5, 50)
    dl_velocity      = rng.gauss(-1.5, 0.5) if label == 1 else rng.gauss(0.0, 1.0)
    script_entropy   = rng.uniform(4.5, 6.5) if label == 1 else rng.uniform(2.0, 4.0)
    dep_count        = rng.randint(0, 5)
    repo_present     = int(rng.random() > (0.7 if label == 1 else 0.1))
    license_present  = int(rng.random() > (0.6 if label == 1 else 0.05))
    has_postinstall  = int(label == 1 and rng.random() > 0.3)
    file_count       = rng.randint(2, 15)
    obfusc_ratio     = rng.uniform(0.2, 0.8) if label == 1 else rng.uniform(0.0, 0.05)
    url_in_code      = rng.randint(1, 5) if label == 1 else rng.randint(0, 1)
    eval_count       = rng.randint(1, 4) if (label == 1 and attack_type == "code_injection") else 0
    b64_count        = rng.randint(1, 3) if (label == 1 and attack_type == "code_injection") else 0
    env_access       = rng.randint(1, 4) if (label == 1 and attack_type == "credential_harvesting") else 0
    net_call_count   = rng.randint(1, 6) if label == 1 else rng.randint(0, 1)
    fs_write_count   = rng.randint(0, 3)
    pkg_size         = rng.uniform(1, 20)
    desc_length      = rng.randint(0, 50) if label == 1 else rng.randint(50, 300)
    keyword_count    = rng.randint(0, 3) if label == 1 else rng.randint(3, 15)
    author_email_ent = rng.uniform(3.0, 5.0) if label == 1 else rng.uniform(1.0, 3.5)
    publish_date_days = rng.randint(0, 30) if label == 1 else rng.randint(100, 2000)

    # Script text snippets (used by rule-based baselines)
    script_fragments: list[str] = []
    if label == 1:
        if attack_type == "credential_harvesting":
            script_fragments += ["os.environ", "requests.post", "subprocess.run"]
        if attack_type == "code_injection":
            script_fragments += ["eval(", "exec(", "base64.b64decode"]
        if attack_type == "typosquatting":
            script_fragments += ["requests.post", "urllib.request.urlopen"]
        if attack_type == "dependency_confusion":
            script_fragments += ["socket.connect", "os.environ"]
        if attack_type == "cryptomining":
            script_fragments += ["hashlib.sha256", "socket.connect"]
        if has_postinstall:
            script_fragments.append("postinstall")

    install_script = " ".join(script_fragments)

    return {
        "name":               name,
        "version":            f"1.{rng.randint(0, 9)}.{rng.randint(0, 9)}",
        "ecosystem":          ecosystem,
        "label":              label,
        "attack_type":        attack_type or "",
        "split":              split,
        # 22 metadata features (matching MetadataExtractor.extract ordering)
        "name_dist":          name_dist,
        "maint_age":          round(maint_age, 1),
        "version_count":      version_count,
        "dl_velocity":        round(dl_velocity, 3),
        "script_entropy":     round(script_entropy, 3),
        "dep_count":          dep_count,
        "repo_present":       repo_present,
        "license_present":    license_present,
        "has_postinstall":    has_postinstall,
        "file_count":         file_count,
        "obfusc_ratio":       round(obfusc_ratio, 3),
        "url_in_code":        url_in_code,
        "eval_count":         eval_count,
        "b64_count":          b64_count,
        "env_access":         env_access,
        "net_call_count":     net_call_count,
        "fs_write_count":     fs_write_count,
        "pkg_size":           round(pkg_size, 2),
        "desc_length":        desc_length,
        "keyword_count":      keyword_count,
        "author_email_entropy": round(author_email_ent, 3),
        "publish_date_days":  publish_date_days,
        # script text for rule-based baselines
        "install_script":     install_script,
    }


META_FEATURES = [
    "name_dist", "maint_age", "version_count", "dl_velocity",
    "script_entropy", "dep_count", "repo_present", "license_present",
    "has_postinstall", "file_count", "obfusc_ratio", "url_in_code",
    "eval_count", "b64_count", "env_access", "net_call_count",
    "fs_write_count", "pkg_size", "desc_length", "keyword_count",
    "author_email_entropy", "publish_date_days",
]


def generate_synthetic_dataset(
    n_malicious: int = 500,
    n_benign: int = 1894,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Generate synthetic train + test rows when real data is unavailable."""
    rng = random.Random(seed)

    def _make_split(n_mal: int, n_ben: int, split: str) -> list[dict]:
        rows = []
        per_type = max(1, n_mal // len(_ATTACK_TYPES))
        idx = 0
        for at in _ATTACK_TYPES:
            for _ in range(per_type):
                rows.append(_generate_row(rng, 1, at, idx, split))
                idx += 1
        for _ in range(n_ben):
            rows.append(_generate_row(rng, 0, None, idx, split))
            idx += 1
        rng.shuffle(rows)
        return rows

    train_rows = _make_split(int(n_malicious * 0.8), int(n_benign * 0.8), "train")
    test_rows  = _make_split(int(n_malicious * 0.2), int(n_benign * 0.2), "test")
    return train_rows, test_rows


# ---------------------------------------------------------------------------
# CSV loading with synthetic fallback
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def cast_row(row: dict) -> dict:
    """Cast numeric string fields to float/int where needed."""
    for f in META_FEATURES + ["label"]:
        if f in row:
            try:
                row[f] = float(row[f])
            except (ValueError, TypeError):
                row[f] = 0.0
    return row


def get_datasets() -> tuple[list[dict], list[dict]]:
    train_rows = [cast_row(r) for r in load_csv(TRAIN_CSV)]
    test_rows  = [cast_row(r) for r in load_csv(TEST_CSV)]

    if not train_rows or not test_rows:
        log.warning(
            "Real dataset CSVs not found under %s — using synthetic data for demonstration.",
            DATA_DIR,
        )
        tr, te = generate_synthetic_dataset()
        if not train_rows:
            train_rows = tr
        if not test_rows:
            test_rows  = te

    return train_rows, test_rows


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return tp, fp, tn, fn


def precision_recall_f1(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[float, float, float]:
    tp, fp, _tn, fn = _confusion(y_true, y_pred)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def fpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    _tp, fp, tn, _fn = _confusion(y_true, y_pred)
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Trapezoidal AUC without sklearn."""
    pairs = sorted(zip(scores, y_true), key=lambda x: -x[0])
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tp_c = fp_c = 0
    prev_fpr = prev_tpr = 0.0
    auc = 0.0
    for _, label in pairs:
        if label == 1:
            tp_c += 1
        else:
            fp_c += 1
        curr_tpr = tp_c / n_pos
        curr_fpr = fp_c / n_neg
        auc += (curr_fpr - prev_fpr) * (curr_tpr + prev_tpr) / 2.0
        prev_tpr, prev_fpr = curr_tpr, curr_fpr
    return auc


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
    n_iter: int = 5000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Bootstrap 95% CI for P, R, F1, FPR, AUC."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    metrics = {"precision": [], "recall": [], "f1": [], "fpr": [], "auc": []}

    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        yt, yp, ys = y_true[idx], y_pred[idx], scores[idx]
        p, r, f = precision_recall_f1(yt, yp)
        metrics["precision"].append(p)
        metrics["recall"].append(r)
        metrics["f1"].append(f)
        metrics["fpr"].append(fpr(yt, yp))
        metrics["auc"].append(roc_auc(yt, ys))

    lo, hi = alpha / 2, 1 - alpha / 2
    ci = {}
    for k, vals in metrics.items():
        arr = np.sort(vals)
        ci[k] = {
            "mean": float(np.mean(arr)),
            "ci_lo": float(arr[int(lo * n_iter)]),
            "ci_hi": float(arr[int(hi * n_iter)]),
        }
    return ci


def compute_full_metrics(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
    n_bootstrap: int = 5000,
) -> dict:
    p, r, f1_val = precision_recall_f1(y_true, y_pred)
    fp_rate = fpr(y_true, y_pred)
    auc_val = roc_auc(y_true, scores)
    ci = bootstrap_ci(y_true, y_pred, scores, n_iter=n_bootstrap)
    return {
        "system": name,
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1_val, 4),
        "fpr": round(fp_rate, 4),
        "auc": round(auc_val, 4),
        "bootstrap_ci": ci,
    }


# ---------------------------------------------------------------------------
# Baseline 1: GuardDog-style rule-based
# ---------------------------------------------------------------------------

_GUARDDOG_RULES = {
    "network_in_setup":  re.compile(
        r"(requests\.(get|post)|urllib\.request|http\.client|socket\.connect|fetch\()"
    ),
    "eval_exec":         re.compile(r"\b(eval|exec)\s*\("),
    "base64_decode":     re.compile(
        r"(base64\.b64decode|atob\s*\(|Buffer\.from.*base64)"
    ),
    "env_exfil":         re.compile(
        r"(os\.environ|process\.env|System\.getenv)"
    ),
    "obfuscated_string": re.compile(
        r"(\\x[0-9a-fA-F]{2}){4,}|([A-Za-z0-9+/]{40,}={0,2})"
    ),
}


def guarddog_score(row: dict) -> float:
    script = str(row.get("install_script", ""))
    fires = sum(1 for r in _GUARDDOG_RULES.values() if r.search(script))
    return float(min(fires, 1))


# ---------------------------------------------------------------------------
# Baseline 2: MalOSS-style (metadata + logistic regression)
# ---------------------------------------------------------------------------

_MALOSS_FEATURES = [
    "name_dist",       # low value = typosquat-like
    "maint_age",       # low = new account
    "repo_present",    # 0 = no repo
    "has_postinstall", # postinstall hook
    "net_call_count",  # network calls
]


def _extract_maloss_features(rows: list[dict]) -> np.ndarray:
    X = np.zeros((len(rows), len(_MALOSS_FEATURES)), dtype=np.float32)
    for i, row in enumerate(rows):
        for j, feat in enumerate(_MALOSS_FEATURES):
            try:
                X[i, j] = float(row.get(feat, 0))
            except (ValueError, TypeError):
                X[i, j] = 0.0
    return X


class LogisticReg:
    """Minimal logistic regression (no sklearn dependency needed)."""

    def __init__(self, lr: float = 0.1, n_iter: int = 1000, reg: float = 0.01):
        self.lr = lr
        self.n_iter = n_iter
        self.reg = reg
        self.w: np.ndarray = np.array([])
        self.b: float = 0.0

    def _sig(self, z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticReg":
        n, d = X.shape
        self.w = np.zeros(d, dtype=np.float64)
        self.b = 0.0
        for _ in range(self.n_iter):
            z = X @ self.w + self.b
            p = self._sig(z)
            err = p - y
            grad_w = (X.T @ err) / n + self.reg * self.w
            grad_b = err.mean()
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._sig(X @ self.w + self.b)


def _std_scale(X_tr: np.ndarray, X_te: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu  = X_tr.mean(axis=0)
    std = X_tr.std(axis=0) + 1e-9
    return (X_tr - mu) / std, (X_te - mu) / std


def maloss_predict(
    train_rows: list[dict], test_rows: list[dict]
) -> tuple[np.ndarray, np.ndarray]:
    X_tr = _extract_maloss_features(train_rows)
    y_tr = np.array([float(r.get("label", 0)) for r in train_rows])
    X_te = _extract_maloss_features(test_rows)

    X_tr_s, X_te_s = _std_scale(X_tr, X_te)
    clf = LogisticReg(lr=0.1, n_iter=2000, reg=0.01).fit(X_tr_s, y_tr)
    scores = clf.predict_proba(X_te_s)
    preds  = (scores >= 0.5).astype(int)
    return preds, scores


# ---------------------------------------------------------------------------
# Baseline 3: CrossLang-style
# ---------------------------------------------------------------------------

_CROSSLANG_PATTERNS = re.compile(
    r"(subprocess.*node|child_process.*python|os\.system.*node"
    r"|exec.*bash|require.*child_process|spawn.*python)"
)
_CROSSLANG_META = ["net_call_count", "eval_count", "b64_count", "has_postinstall", "url_in_code"]


def _extract_crosslang_features(rows: list[dict]) -> np.ndarray:
    X = np.zeros((len(rows), len(_CROSSLANG_META) + 1), dtype=np.float32)
    for i, row in enumerate(rows):
        script = str(row.get("install_script", ""))
        for j, feat in enumerate(_CROSSLANG_META):
            try:
                X[i, j] = float(row.get(feat, 0))
            except (ValueError, TypeError):
                X[i, j] = 0.0
        X[i, -1] = float(bool(_CROSSLANG_PATTERNS.search(script)))
    return X


def crosslang_predict(
    train_rows: list[dict], test_rows: list[dict]
) -> tuple[np.ndarray, np.ndarray]:
    X_tr = _extract_crosslang_features(train_rows)
    y_tr = np.array([float(r.get("label", 0)) for r in train_rows])
    X_te = _extract_crosslang_features(test_rows)

    X_tr_s, X_te_s = _std_scale(X_tr, X_te)
    clf = LogisticReg(lr=0.1, n_iter=2000, reg=0.01).fit(X_tr_s, y_tr)
    scores = clf.predict_proba(X_te_s)
    preds  = (scores >= 0.5).astype(int)
    return preds, scores


# ---------------------------------------------------------------------------
# Baseline 4: Metadata-only (all 22 features)
# ---------------------------------------------------------------------------

def _extract_meta22(rows: list[dict]) -> np.ndarray:
    X = np.zeros((len(rows), len(META_FEATURES)), dtype=np.float32)
    for i, row in enumerate(rows):
        for j, feat in enumerate(META_FEATURES):
            try:
                X[i, j] = float(row.get(feat, 0))
            except (ValueError, TypeError):
                X[i, j] = 0.0
    return X


def meta_only_predict(
    train_rows: list[dict], test_rows: list[dict]
) -> tuple[np.ndarray, np.ndarray]:
    X_tr = _extract_meta22(train_rows)
    y_tr = np.array([float(r.get("label", 0)) for r in train_rows])
    X_te = _extract_meta22(test_rows)

    X_tr_s, X_te_s = _std_scale(X_tr, X_te)
    clf = LogisticReg(lr=0.1, n_iter=2000, reg=0.01).fit(X_tr_s, y_tr)
    scores = clf.predict_proba(X_te_s)
    preds  = (scores >= 0.5).astype(int)
    return preds, scores


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

PRED_FIELDS = ["name", "ecosystem", "version", "label", "attack_type",
               "score", "prediction"]


def save_predictions(
    baseline_name: str,
    test_rows: list[dict],
    preds: np.ndarray,
    scores: np.ndarray,
) -> Path:
    out_path = BASELINES_DIR / f"{baseline_name}_predictions.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=PRED_FIELDS)
        writer.writeheader()
        for row, pred, score in zip(test_rows, preds, scores):
            writer.writerow({
                "name":       row.get("name", ""),
                "ecosystem":  row.get("ecosystem", ""),
                "version":    row.get("version", ""),
                "label":      int(float(row.get("label", 0))),
                "attack_type": row.get("attack_type", ""),
                "score":      round(float(score), 6),
                "prediction": int(pred),
            })
    return out_path


def save_metrics(baseline_name: str, metrics: dict) -> Path:
    out_path = BASELINES_DIR / f"{baseline_name}_metrics.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM-Sentry baseline runner")
    p.add_argument(
        "--bootstrap-iter", type=int, default=5000,
        help="Number of bootstrap iterations for CI (default: 5000)."
    )
    p.add_argument(
        "--seed", type=int, default=42, help="Random seed."
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading datasets...")
    train_rows, test_rows = get_datasets()
    log.info("Train: %d rows | Test: %d rows", len(train_rows), len(test_rows))

    y_true = np.array([float(r.get("label", 0)) for r in test_rows])

    results_summary: list[dict] = []

    # ------------------------------------------------------------------
    # 1. GuardDog-style
    # ------------------------------------------------------------------
    log.info("Running Baseline 1: GuardDog-style rule-based...")
    gd_scores = np.array([guarddog_score(r) for r in test_rows])
    gd_preds  = (gd_scores >= 0.5).astype(int)
    save_predictions("guarddog", test_rows, gd_preds, gd_scores)
    gd_metrics = compute_full_metrics(
        "GuardDog", y_true, gd_preds, gd_scores, n_bootstrap=args.bootstrap_iter
    )
    save_metrics("guarddog", gd_metrics)
    results_summary.append(gd_metrics)
    log.info(
        "GuardDog — F1: %.4f | FPR: %.4f | AUC: %.4f",
        gd_metrics["f1"], gd_metrics["fpr"], gd_metrics["auc"],
    )

    # ------------------------------------------------------------------
    # 2. MalOSS-style
    # ------------------------------------------------------------------
    log.info("Running Baseline 2: MalOSS-style metadata classifier...")
    mo_preds, mo_scores = maloss_predict(train_rows, test_rows)
    save_predictions("maloss", test_rows, mo_preds, mo_scores)
    mo_metrics = compute_full_metrics(
        "MalOSS", y_true, mo_preds, mo_scores, n_bootstrap=args.bootstrap_iter
    )
    save_metrics("maloss", mo_metrics)
    results_summary.append(mo_metrics)
    log.info(
        "MalOSS — F1: %.4f | FPR: %.4f | AUC: %.4f",
        mo_metrics["f1"], mo_metrics["fpr"], mo_metrics["auc"],
    )

    # ------------------------------------------------------------------
    # 3. CrossLang-style
    # ------------------------------------------------------------------
    log.info("Running Baseline 3: CrossLang-style classifier...")
    cl_preds, cl_scores = crosslang_predict(train_rows, test_rows)
    save_predictions("crosslang", test_rows, cl_preds, cl_scores)
    cl_metrics = compute_full_metrics(
        "CrossLang", y_true, cl_preds, cl_scores, n_bootstrap=args.bootstrap_iter
    )
    save_metrics("crosslang", cl_metrics)
    results_summary.append(cl_metrics)
    log.info(
        "CrossLang — F1: %.4f | FPR: %.4f | AUC: %.4f",
        cl_metrics["f1"], cl_metrics["fpr"], cl_metrics["auc"],
    )

    # ------------------------------------------------------------------
    # 4. Metadata-only baseline
    # ------------------------------------------------------------------
    log.info("Running Baseline 4: Metadata-only (22 features)...")
    m22_preds, m22_scores = meta_only_predict(train_rows, test_rows)
    save_predictions("meta_only", test_rows, m22_preds, m22_scores)
    m22_metrics = compute_full_metrics(
        "Meta-only", y_true, m22_preds, m22_scores, n_bootstrap=args.bootstrap_iter
    )
    save_metrics("meta_only", m22_metrics)
    results_summary.append(m22_metrics)
    log.info(
        "Meta-only — F1: %.4f | FPR: %.4f | AUC: %.4f",
        m22_metrics["f1"], m22_metrics["fpr"], m22_metrics["auc"],
    )

    # ------------------------------------------------------------------
    # Combined summary
    # ------------------------------------------------------------------
    summary_path = BASELINES_DIR / "baselines_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(results_summary, fh, indent=2)

    log.info("All baselines complete. Summary: %s", summary_path)
    log.info("\n%-14s %6s %6s %6s %6s %6s", "System", "P", "R", "F1", "FPR", "AUC")
    log.info("-" * 50)
    for m in results_summary:
        log.info(
            "%-14s %6.4f %6.4f %6.4f %6.4f %6.4f",
            m["system"], m["precision"], m["recall"],
            m["f1"], m["fpr"], m["auc"],
        )


if __name__ == "__main__":
    main()
