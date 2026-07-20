# Datasets for LLM-Sentry Evaluation

## 1. PyPI Malware Dataset (PMD)

Source: DataDog GuardDog + Socket MSR 2024 Benchmark

Access:
- https://github.com/DataDog/guarddog (open-source tool + malicious sample index)
- https://github.com/ossf/malicious-packages (OpenSSF curated malicious package reports)

Split used in this paper:
- Total: 9,411 packages
- Malicious: 1,870 (19.9%)
- Benign: 7,541 (80.1%)
- Ecosystems: npm, PyPI

## 2. MalwareBazaar-OSS (MB-OSS)

Constructed by merging:
1. ConfuGuard dependency confusion dataset (630 confirmed attacks)
2. MSR 2024 malicious package corpus (Socket database, Q3-Q4 2023)
3. Benign baseline: top-10,000 downloaded packages from PyPI and npm (Q1 2025)

Total: 18,942 packages

Attack category breakdown:
- Typosquatting: 1,124 malicious packages
- Dependency confusion: 782 malicious packages
- Code injection: 1,038 malicious packages
- Credential harvesting: 891 malicious packages
- Crypto-mining: 381 malicious packages

Split: 80/10/10 train/val/test, stratified by attack category.

## 3. Responsible Disclosure

47 previously unreported malicious packages identified during dataset construction were reported to:
- PyPI Security Advisory (security@pypi.org)
- npm Security (security@npmjs.com)

All packages were removed from registries before this paper's submission.

## 4. Access Instructions

Dataset access is provided via the GitHub repository:
https://github.com/ufxa/Software-Supply-Chain-Security

The `data/` directory contains:
- `pmd_labels.csv`: Package name, version, ecosystem, label for PMD split
- `mboss_labels.csv`: Package name, version, ecosystem, label, attack_category for MB-OSS split
- `benign_baseline.csv`: Top-10,000 benign packages sampled from registries
