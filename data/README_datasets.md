# Datasets for LLM-Sentry Evaluation

This directory contains the datasets used in the paper experiments and the scripts to reproduce them.

---

## Files in this directory

### `real_dataset_5963.csv`

A real-world dataset of 5,963 packages collected from public sources:

| Field | Description |
|---|---|
| `name` | Package name |
| `ecosystem` | `npm` or `pypi` |
| `label` | `1` = malicious, `0` = benign |
| `is_deleted` | Whether the package was removed from the registry |
| `num_versions` | Number of published versions |
| `num_maintainers` | Number of maintainers |
| `age_days` | Days since first publish |
| `has_readme` | Whether a README/description is present |
| `has_homepage` | Whether a homepage URL is present |
| `total_downloads` | Total download count (0 for npm; from PyPI registry API) |

**Composition:**
- 2,000 malicious packages — sourced from the [OpenSSF malicious-packages](https://github.com/ossf/malicious-packages) repository (1,000 npm + 1,000 PyPI)
- 3,963 benign packages — top-downloaded packages from npm and PyPI registries

**Note on survival bias:** 85% of the malicious packages have already been removed from registries (`is_deleted=1`). This is intrinsic to the OSSF corpus — the packages were caught and taken down. Evaluation on this set measures detection of *archived* malicious packages, not live ones.

### `ossf_malicious_2000.json`

The raw list of 2,000 malicious package names and ecosystems collected from the OpenSSF malicious-packages GitHub repository. Used as input to generate `real_dataset_5963.csv`.

Format: JSON array of `{"name": "...", "ecosystem": "npm|pypi"}` objects.

---

## How to reproduce the dataset

### Step 1 — Collect the OSSF malicious package list

Requires a GitHub Personal Access Token (raises rate limit from 60 to 5,000 req/hr):

```bash
export GITHUB_TOKEN=ghp_your_token_here
python code/build_ombb25_dataset.py
```

This fetches the full OSSF malicious-packages repository directory listing and saves to `/tmp/ossf_pkg_list.json`.

### Step 2 — Build the metadata dataset

```bash
python code/build_dataset.py
```

Queries npm and PyPI registry APIs for each package and writes metadata CSV.

### Step 3 — Run the experiment

```bash
python code/a100_runner.py
```

Runs the full LLM-Sentry 8-stage pipeline. On CPU: several hours. On A100 GPU: 6-90 minutes depending on dataset size.

---

## About the paper's benchmark datasets

### PyPI Malware Dataset (PMD)

- **Source:** DataDog GuardDog + OpenSSF malicious-packages
- **Total:** 9,411 packages (1,870 malicious / 7,541 benign)
- **Ecosystems:** npm, PyPI
- **Access:** [https://github.com/ossf/malicious-packages](https://github.com/ossf/malicious-packages)

### OSS-MalBench-2025 (OMB-25)

Constructed from:
1. OpenSSF malicious-packages corpus (npm + PyPI)
2. Benign baseline: top-10,000 downloaded packages from PyPI and npm (Q1 2025)

- **Total:** 18,942 packages
- **Split:** 80/10/10 train/val/test, stratified by attack category

Attack category breakdown:
- Typosquatting: 1,124 malicious packages
- Dependency confusion: 782 malicious packages
- Code injection: 1,038 malicious packages
- Credential harvesting: 891 malicious packages
- Crypto-mining: 381 malicious packages

Reproducing the full OMB-25 split requires `GITHUB_TOKEN` set and approximately 4 hours of API calls via `code/build_ombb25_dataset.py`.

---

## Responsible Disclosure

Newly identified malicious packages discovered during dataset construction were reported to:
- PyPI Security Advisory (security@pypi.org)
- npm Security (security@npmjs.com)

All flagged packages were removed from registries before this paper's submission.
