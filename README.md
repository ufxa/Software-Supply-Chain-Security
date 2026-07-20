# LLM-Sentry: Detecting Malicious Packages and Dependency Poisoning in Software Supply Chains

**Paper:** LLM-Sentry: A Large Language Model Framework for Detecting Malicious Packages and Dependency Poisoning in Software Supply Chains

**Author:** Allan Douglas Costa  
**Affiliation:** Federal Rural University of the Amazon (UFRA), Institute of Cyberspace (ICIBE), Belem, Brazil  
**ORCID:** https://orcid.org/0000-0002-7068-8889  
**Contact:** allan.costa@ufra.edu.br  

**Repository:** https://github.com/ufxa/Software-Supply-Chain-Security

---

## Abstract

Software supply chain attacks via public package registries such as npm and PyPI have grown exponentially, with over 1.2 million malicious packages catalogued through 2025. LLM-Sentry is a multi-stage detection framework combining fine-tuned LLMs with static metadata analysis and behavioral sequence modeling. It introduces the **Package Risk Confidence Score (PRCS)**, a novel composite metric achieving F1 = 0.961 on the OMB-25 benchmark of 18,942 labeled packages.

---

## Repository Structure

```
.
├── manuscript/
│   ├── main.tex                  # IEEE LaTeX source
│   └── references.bib            # BibTeX bibliography (40 verified references)
├── code/
│   ├── llm_sentry.py             # Main detection framework
│   ├── scan_oss_repos.py         # OSS registry scanning pipeline
│   ├── evaluate.py               # Evaluation metrics and CI computation
│   └── requirements.txt          # Python dependencies
├── data/
│   └── README_datasets.md        # Dataset access instructions
├── figures/                      # Generated figures (via LaTeX/pgfplots)
├── research/                     # Research notes and literature synthesis
├── reviews/                      # Peer review materials
└── README.md
```

---

## Datasets

### PyPI Malware Dataset (PMD)
- Source: Socket internal database + DataDog GuardDog (MSR 2024)
- Packages: 9,411 (1,870 malicious, 7,541 benign)
- Access: https://github.com/DataDog/guarddog

### OSS-MalBench-2025 (OMB-25)
- Constructed from: ConfuGuard dataset + MSR 2024 corpus + registry sampling
- Packages: 18,942 (4,216 malicious, 14,726 benign)
- Attack categories: typosquatting, dependency confusion, code injection, credential harvesting, crypto-mining
- Access: See `data/README_datasets.md`

---

## Installation

```bash
pip install -r code/requirements.txt
```

Set your OpenAI API key:
```bash
export OPENAI_API_KEY=your-key-here
```

---

## Usage

### Analyze a single package

```bash
python code/llm_sentry.py /path/to/package.tar.gz
```

Example output:
```json
{
  "archive": "suspicious-package-1.0.0.tar.gz",
  "prcs": 0.8731,
  "label": "MALICIOUS",
  "s_meta": 0.712,
  "s_sem": 0.941,
  "s_beh": 0.882,
  "attack_type": "credential_harvesting",
  "risk_indicators": ["eval() in install script", "network call to external IP"],
  "llm_reasoning": "The install script contains an obfuscated eval() call that..."
}
```

### Bulk scan from registry

```bash
python code/scan_oss_repos.py --input-json packages.json --output-csv results.csv
```

### Evaluate predictions

```bash
python code/evaluate.py results.csv --output eval_results.json
```

---

## PRCS Metric

The Package Risk Confidence Score (PRCS) is defined as:

$$\text{PRCS}(p) = w_1 \cdot s_{\text{meta}}(p) + w_2 \cdot s_{\text{sem}}(p) + w_3 \cdot s_{\text{beh}}(p)$$

with optimal weights $[w_1, w_2, w_3] = [0.25, 0.50, 0.25]$ and decision threshold $\tau = 0.55$.

---

## Results

| System | F1 | FPR | AUC |
|--------|----|-----|-----|
| MalOSS | 0.850 | 0.074 | 0.921 |
| DONAPI | 0.913 | 0.038 | 0.962 |
| BERD | 0.934 | 0.028 | 0.978 |
| **LLM-Sentry** | **0.961** | **0.012** | **0.991** |

---

## Citation

```bibtex
@inproceedings{costa2025llmsentry,
  author    = {Costa, Allan Douglas},
  title     = {{LLM-Sentry}: A Large Language Model Framework for Detecting
               Malicious Packages and Dependency Poisoning in Software Supply Chains},
  booktitle = {Proc. IEEE International Conference on ...},
  year      = {2025},
  publisher = {IEEE}
}
```

---

## License

MIT License. See LICENSE file for details.

---

## Acknowledgments

This work was supported by FAPESPA, PRODEPA, the Government of the State of Para, SEC365, LICA/UFRA, CCAD-IA/UFPA, RNP, and INCT iAmazonia.
