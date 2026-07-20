"""
evaluate.py -- Evaluation script for LLM-Sentry on MB-OSS and PMD benchmarks.
Author: Allan Douglas Costa (UFRA / LICA / SEC365)
Paper: "LLM-Sentry: A Large Language Model Framework for Detecting
        Malicious Packages and Dependency Poisoning in Software Supply Chains"
Repository: https://github.com/ufxa/Software-Supply-Chain-Security
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

def precision_recall_f1(y_true: Sequence[int],
                         y_pred: Sequence[int]) -> tuple[float, float, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    p = tp / (tp + fp + 1e-9)
    r = tp / (tp + fn + 1e-9)
    f1 = 2 * p * r / (p + r + 1e-9)
    return p, r, f1


def false_positive_rate(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    return fp / (fp + tn + 1e-9)


def wilson_ci(count: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if n == 0:
        return 0.0, 0.0
    p_hat = count / n
    center = (p_hat + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
    margin = (z * math.sqrt(p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2))
              / (1 + z ** 2 / n))
    return max(0.0, center - margin), min(1.0, center + margin)


def bootstrap_f1_ci(y_true: list[int], y_pred: list[int],
                    n_boot: int = 1000,
                    seed: int = 42) -> tuple[float, float]:
    """Bootstrap 95% CI for F1 score."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    f1_scores = []
    yt_arr = np.array(y_true)
    yp_arr = np.array(y_pred)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        _, _, f1 = precision_recall_f1(yt_arr[idx].tolist(), yp_arr[idx].tolist())
        f1_scores.append(f1)
    lo, hi = np.percentile(f1_scores, [2.5, 97.5])
    return float(lo), float(hi)


# ------------------------------------------------------------------
# Main evaluation function
# ------------------------------------------------------------------

def evaluate(results_csv: str, output_json: str = "eval_results.json") -> dict:
    """
    Evaluate predictions in a CSV file and compute all metrics.
    Expected CSV columns: true_label, pred_label
    """
    import csv

    y_true, y_pred = [], []
    with open(results_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("error"):
                continue
            try:
                y_true.append(int(row["true_label"]))
                y_pred.append(int(row["pred_label"]))
            except (KeyError, ValueError):
                pass

    n = len(y_true)
    p, r, f1 = precision_recall_f1(y_true, y_pred)
    fpr = false_positive_rate(y_true, y_pred)
    tp = sum(1 for t, p_ in zip(y_true, y_pred) if t == 1 and p_ == 1)

    f1_lo, f1_hi = bootstrap_f1_ci(y_true, y_pred)
    fpr_lo, fpr_hi = wilson_ci(
        sum(1 for t, p_ in zip(y_true, y_pred) if t == 0 and p_ == 1),
        sum(1 for t in y_true if t == 0)
    )

    results = {
        "n_packages": n,
        "n_malicious_true": sum(y_true),
        "n_benign_true": n - sum(y_true),
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "f1_ci_95": [round(f1_lo, 4), round(f1_hi, 4)],
        "fpr": round(fpr, 4),
        "fpr_ci_95": [round(fpr_lo, 4), round(fpr_hi, 4)],
    }

    print(json.dumps(results, indent=2))
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM-Sentry Evaluation")
    parser.add_argument("results_csv", help="CSV file with true_label and pred_label columns")
    parser.add_argument("--output", default="eval_results.json")
    args = parser.parse_args()
    evaluate(args.results_csv, args.output)
