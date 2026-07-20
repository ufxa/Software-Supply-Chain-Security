"""
LLM-Sentry: Multi-Stage Malicious Package Detection Framework
Author: Allan Douglas Costa (UFRA / LICA / SEC365)
Paper: "LLM-Sentry: A Large Language Model Framework for Detecting
        Malicious Packages and Dependency Poisoning in Software Supply Chains"
Repository: https://github.com/ufxa/Software-Supply-Chain-Security
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import tarfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from openai import OpenAI

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

PRCS_WEIGHTS = [0.25, 0.50, 0.25]   # [w_meta, w_sem, w_beh]
PRCS_THRESHOLD = 0.55
GAMMA = 0.45                          # CodeBERT vs LLM fusion weight
ALPHA = 0.30                          # Behavioral network-call weight

MALICIOUS_PATTERNS = {
    "data_exfiltration": ["requests.post", "urllib.request.urlopen", "socket.connect"],
    "credential_harvesting": ["os.environ", "subprocess.run", "getpass.getpass"],
    "persistence": ["crontab", "registry", "autostart"],
    "crypto_mining": ["hashlib.sha256", "mining", "wallet_address"],
    "reverse_shell": ["socket.socket", "os.dup2", "pty.spawn"],
}

LLM_PROMPT_SYSTEM = (
    "You are a supply chain security analyst. "
    "Analyze the following package for malicious indicators. "
    'Respond only with JSON: {"risk_indicators": [str], "attack_type": str or null, '
    '"confidence": float, "reasoning": str}'
)


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class PackageFeatures:
    """Feature container for a single package analysis."""

    name: str
    version: str
    ecosystem: str

    # Stage 2 - Metadata
    name_dist: int = 0
    maint_age: float = 0.0
    version_count: int = 0
    dl_velocity: float = 0.0
    script_entropy: float = 0.0
    dep_count: int = 0
    repo_present: bool = False
    license_present: bool = False
    has_postinstall: bool = False
    file_count: int = 0
    obfusc_ratio: float = 0.0
    url_in_code: int = 0
    eval_count: int = 0
    b64_count: int = 0
    env_access: int = 0
    net_call_count: int = 0
    fs_write_count: int = 0
    pkg_size: float = 0.0
    desc_length: int = 0
    keyword_count: int = 0
    author_email_entropy: float = 0.0
    publish_date_days: int = 0

    # Stage 3 - Code embedding (768-dim)
    code_embedding: Optional[np.ndarray] = field(default=None, repr=False)

    # Stage 4 - LLM output
    llm_confidence: float = 0.0
    llm_attack_type: Optional[str] = None
    llm_risk_indicators: list[str] = field(default_factory=list)
    llm_reasoning: str = ""

    # Stage 5 - Behavioral
    api_sequence: list[str] = field(default_factory=list)
    behavioral_deviation: float = 0.0

    # Stage 6 - PRCS
    s_meta: float = 0.0
    s_sem: float = 0.0
    s_beh: float = 0.0
    prcs: float = 0.0
    label: int = 0  # 1=malicious, 0=benign


@dataclass
class PRCSResult:
    """Output of the PRCS computation stage."""

    prcs: float
    label: int
    s_meta: float
    s_sem: float
    s_beh: float
    llm_reasoning: str
    attack_type: Optional[str]
    risk_indicators: list[str]


# ------------------------------------------------------------------
# Stage 1: Ingestion
# ------------------------------------------------------------------

class PackageIngestionModule:
    """Decompress and normalize package archives."""

    SUPPORTED_EXTENSIONS = (".tar.gz", ".whl", ".tgz", ".zip")

    def ingest(self, archive_path: str) -> dict:
        """Return a virtual filesystem dict {relative_path: bytes}."""
        archive_path = Path(archive_path)
        vfs: dict[str, bytes] = {}

        if archive_path.suffix == ".whl" or archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zf:
                for name in zf.namelist():
                    vfs[name] = zf.read(name)
        else:
            with tarfile.open(archive_path, "r:gz") as tf:
                for member in tf.getmembers():
                    if member.isfile():
                        f = tf.extractfile(member)
                        if f:
                            vfs[member.name] = f.read()

        return vfs

    def extract_install_scripts(self, vfs: dict) -> list[str]:
        """Extract install-time scripts content."""
        install_keys = ["setup.py", "package.json", "install.js",
                        "postinstall.sh", "preinstall.sh"]
        scripts = []
        for key, content in vfs.items():
            if any(key.endswith(k) for k in install_keys):
                try:
                    scripts.append(content.decode("utf-8", errors="replace"))
                except Exception:
                    pass
        return scripts

    def extract_manifest(self, vfs: dict) -> dict:
        """Parse package manifest (package.json or setup.py)."""
        for key, content in vfs.items():
            if key.endswith("package.json") and "/" not in key.lstrip("./"):
                try:
                    return json.loads(content.decode("utf-8"))
                except Exception:
                    pass
        return {}


# ------------------------------------------------------------------
# Stage 2: Metadata Feature Extraction
# ------------------------------------------------------------------

class MetadataExtractor:
    """Compute 22-dimensional metadata feature vector."""

    TOP_POPULAR_PACKAGES = [
        "requests", "numpy", "pandas", "flask", "django", "scipy",
        "matplotlib", "tensorflow", "torch", "sklearn", "express",
        "lodash", "react", "axios", "moment", "webpack", "babel",
        "jest", "eslint", "typescript", "vue", "angular",
    ]

    def levenshtein(self, a: str, b: str) -> int:
        """Compute Levenshtein distance between two strings."""
        if len(a) < len(b):
            return self.levenshtein(b, a)
        if not b:
            return len(a)
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a):
            curr = [i + 1]
            for j, cb in enumerate(b):
                curr.append(min(prev[j + 1] + 1, curr[-1] + 1,
                                prev[j] + (ca != cb)))
            prev = curr
        return prev[-1]

    def compute_shannon_entropy(self, text: str) -> float:
        """Shannon entropy of a string (bits per character)."""
        if not text:
            return 0.0
        freq = {}
        for c in text:
            freq[c] = freq.get(c, 0) + 1
        n = len(text)
        return -sum((f / n) * math.log2(f / n) for f in freq.values())

    def extract(self, manifest: dict, vfs: dict,
                install_scripts: list[str]) -> np.ndarray:
        """Return a 22-dimensional metadata feature vector."""
        name = manifest.get("name", "")
        min_dist = min(
            (self.levenshtein(name.lower(), pkg) for pkg in self.TOP_POPULAR_PACKAGES),
            default=99,
        )

        scripts_concat = "\n".join(install_scripts)
        url_pattern = re.compile(r'https?://[\w./-]+')
        eval_pattern = re.compile(r'\b(eval|exec|Function)\s*\(')
        b64_pattern = re.compile(r'(atob|Buffer\.from.*base64|base64\.b64decode)')
        env_pattern = re.compile(r'(process\.env|os\.environ|System\.getenv)')
        net_pattern = re.compile(
            r'(fetch|axios|http\.|https\.|urllib|requests\.|socket\.connect)')
        fs_pattern = re.compile(
            r'(fs\.write|open\s*\(|shutil\.|WriteFile|createWriteStream)')

        features = np.array([
            float(min_dist),                                    # name_dist
            float(manifest.get("_maintainer_age_days", 365)),  # maint_age
            float(manifest.get("_version_count", 1)),           # version_count
            float(manifest.get("_dl_velocity_zscore", 0.0)),    # dl_velocity
            float(self.compute_shannon_entropy(scripts_concat)), # script_entropy
            float(len(manifest.get("dependencies", {}))),       # dep_count
            float(bool(manifest.get("repository"))),             # repo_present
            float(bool(manifest.get("license"))),                # license_present
            float(bool(manifest.get("scripts", {}).get("postinstall"))),  # has_postinstall
            float(len(vfs)),                                     # file_count
            float(manifest.get("_obfusc_ratio", 0.0)),          # obfusc_ratio
            float(len(url_pattern.findall(scripts_concat))),     # url_in_code
            float(len(eval_pattern.findall(scripts_concat))),    # eval_count
            float(len(b64_pattern.findall(scripts_concat))),     # b64_count
            float(len(env_pattern.findall(scripts_concat))),     # env_access
            float(len(net_pattern.findall(scripts_concat))),     # net_call_count
            float(len(fs_pattern.findall(scripts_concat))),      # fs_write_count
            float(sum(len(v) for v in vfs.values()) / 1024),    # pkg_size (KB)
            float(len(manifest.get("description", ""))),         # desc_length
            float(len(manifest.get("keywords", []))),            # keyword_count
            float(self.compute_shannon_entropy(
                manifest.get("author", {}).get("email", "") if isinstance(
                    manifest.get("author"), dict) else manifest.get("author", "")
            )),                                                   # author_email_entropy
            float(manifest.get("_publish_date_days", 0)),        # publish_date_days
        ], dtype=np.float32)

        return features


# ------------------------------------------------------------------
# Stage 3: Static AST and Code Embedding (CodeBERT stub)
# ------------------------------------------------------------------

class CodeEmbeddingModule:
    """
    Produce 768-dimensional code embeddings using CodeBERT.
    Requires: transformers, torch
    """

    def __init__(self, model_name: str = "microsoft/codebert-base"):
        self._model = None
        self._tokenizer = None
        self._model_name = model_name

    def _load_model(self):
        from transformers import AutoModel, AutoTokenizer
        import torch
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModel.from_pretrained(self._model_name)
        self._model.eval()

    def embed(self, vfs: dict) -> np.ndarray:
        """Return mean-pooled 768-dim code embedding across all source files."""
        if self._model is None:
            self._load_model()

        import torch

        source_texts = []
        weights = []
        for path, content in vfs.items():
            if path.endswith((".py", ".js", ".ts")):
                try:
                    text = content.decode("utf-8", errors="replace")[:1024]
                    source_texts.append(text)
                    weights.append(len(content))
                except Exception:
                    pass

        if not source_texts:
            return np.zeros(768, dtype=np.float32)

        embeddings = []
        with torch.no_grad():
            for text in source_texts:
                inputs = self._tokenizer(
                    text, return_tensors="pt",
                    max_length=512, truncation=True, padding=True
                )
                outputs = self._model(**inputs)
                emb = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
                embeddings.append(emb)

        weights_arr = np.array(weights[:len(embeddings)], dtype=np.float32)
        weights_arr /= weights_arr.sum() + 1e-9
        return np.average(embeddings, axis=0, weights=weights_arr)

    def extract_suspicious_ast_nodes(self, vfs: dict) -> list[str]:
        """Return list of suspicious AST subtree repr strings from Python files."""
        suspicious = []
        for path, content in vfs.items():
            if not path.endswith(".py"):
                continue
            try:
                tree = ast.parse(content.decode("utf-8", errors="replace"))
            except Exception:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id in ("eval", "exec"):
                        suspicious.append(ast.unparse(node)[:200])
                    elif isinstance(node.func, ast.Attribute):
                        attr = node.func.attr
                        if attr in ("system", "popen", "connect", "urlopen"):
                            suspicious.append(ast.unparse(node)[:200])
        return suspicious[:10]


# ------------------------------------------------------------------
# Stage 4: LLM Semantic Analysis
# ------------------------------------------------------------------

class LLMAnalyzer:
    """Query GPT-4o (or compatible) to reason about package maliciousness."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-2024-11-20"):
        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self._model = model

    def analyze(self, manifest: dict, install_scripts: list[str],
                 suspicious_nodes: list[str]) -> dict:
        """Return LLM analysis dict with confidence, attack_type, reasoning."""
        name = manifest.get("name", "unknown")
        version = manifest.get("version", "0.0.0")
        ecosystem = manifest.get("_ecosystem", "npm")

        meta_summary = json.dumps({
            "name": name, "version": version,
            "dependencies": list(manifest.get("dependencies", {}).keys())[:10],
            "has_postinstall": bool(manifest.get("scripts", {}).get("postinstall")),
            "description": manifest.get("description", "")[:200],
        }, indent=2)

        script_excerpt = "\n---\n".join(install_scripts)[:2000]
        ast_excerpt = "\n".join(suspicious_nodes)[:500]

        user_message = (
            f"Package: {name} v{version} ({ecosystem})\n"
            f"Metadata:\n{meta_summary}\n\n"
            f"Install script (truncated):\n{script_excerpt}\n\n"
            f"Suspicious AST nodes:\n{ast_excerpt}"
        )

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": LLM_PROMPT_SYSTEM},
                    {"role": "user", "content": user_message},
                ],
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)
            return {
                "confidence": float(result.get("confidence", 0.0)),
                "attack_type": result.get("attack_type"),
                "risk_indicators": result.get("risk_indicators", []),
                "reasoning": result.get("reasoning", ""),
            }
        except Exception as exc:
            return {"confidence": 0.0, "attack_type": None,
                    "risk_indicators": [], "reasoning": str(exc)}


# ------------------------------------------------------------------
# Stage 5: Behavioral Sequence Analysis
# ------------------------------------------------------------------

class BehavioralAnalyzer:
    """Execute install scripts in a sandboxed environment and analyze API calls."""

    def _flatten_patterns(self) -> set[str]:
        flat = set()
        for patterns in MALICIOUS_PATTERNS.values():
            flat.update(patterns)
        return flat

    def analyze_static(self, install_scripts: list[str]) -> tuple[list[str], float]:
        """
        Static behavioral analysis (fallback when sandbox is unavailable).
        Returns (api_sequence, b_dev).
        """
        all_calls = []
        net_calls = 0
        for script in install_scripts:
            for pattern_group in MALICIOUS_PATTERNS.values():
                for pat in pattern_group:
                    if pat in script:
                        all_calls.append(pat)
                        if any(net_kw in pat for net_kw in
                               ["socket", "urllib", "requests", "http", "fetch"]):
                            net_calls += 1

        known_patterns = self._flatten_patterns()
        total = max(len(all_calls), 1)
        intersection = len(set(all_calls) & known_patterns)
        b_dev = intersection / len(known_patterns) + ALPHA * net_calls / total
        return all_calls, min(1.0, b_dev)


# ------------------------------------------------------------------
# Stage 6: PRCS Computation
# ------------------------------------------------------------------

class PRCSEngine:
    """Compute the Package Risk Confidence Score (PRCS)."""

    def __init__(self,
                 weights: list[float] = PRCS_WEIGHTS,
                 threshold: float = PRCS_THRESHOLD,
                 gamma: float = GAMMA):
        assert len(weights) == 3 and abs(sum(weights) - 1.0) < 1e-6
        self.w = weights
        self.tau = threshold
        self.gamma = gamma
        # Logistic regression coefficients (pre-trained on training set)
        self._beta = None
        self._beta0 = 0.0
        # Malicious centroid in embedding space (768-dim)
        self._mu_mal = None

    def _sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
        return float(np.dot(a, b) / denom)

    def compute_s_meta(self, meta_features: np.ndarray) -> float:
        """Eq. (2): logistic regression on metadata features."""
        if self._beta is None:
            # Default: use normalized sum as proxy (replaced by trained model)
            score = float(np.tanh(np.mean(meta_features) * 0.5))
            return max(0.0, min(1.0, score))
        return self._sigmoid(float(self._beta @ meta_features) + self._beta0)

    def compute_s_sem(self, code_embedding: np.ndarray,
                      llm_confidence: float) -> float:
        """Eq. (3): fusion of CodeBERT cosine similarity and LLM confidence."""
        if self._mu_mal is None or code_embedding is None:
            return llm_confidence
        cos = self._cosine_sim(code_embedding, self._mu_mal)
        cos_norm = (cos + 1.0) / 2.0  # map [-1,1] -> [0,1]
        return self.gamma * cos_norm + (1.0 - self.gamma) * llm_confidence

    def compute_s_beh(self, b_dev: float) -> float:
        """Eq. (4): clamped behavioral deviation score."""
        return min(1.0, b_dev)

    def compute(self, features: PackageFeatures) -> PRCSResult:
        """Compute PRCS and return a PRCSResult."""
        meta_vec = np.array([
            features.name_dist, features.maint_age, features.version_count,
            features.dl_velocity, features.script_entropy, features.dep_count,
            float(features.repo_present), float(features.license_present),
            float(features.has_postinstall), features.file_count,
            features.obfusc_ratio, features.url_in_code, features.eval_count,
            features.b64_count, features.env_access, features.net_call_count,
            features.fs_write_count, features.pkg_size, features.desc_length,
            features.keyword_count, features.author_email_entropy,
            features.publish_date_days,
        ], dtype=np.float32)

        # Normalize each feature to [0,1] using known max values
        meta_norm = np.clip(meta_vec / (np.abs(meta_vec).max() + 1e-9), 0, 1)

        s_meta = self.compute_s_meta(meta_norm)
        s_sem = self.compute_s_sem(features.code_embedding, features.llm_confidence)
        s_beh = self.compute_s_beh(features.behavioral_deviation)

        prcs = self.w[0] * s_meta + self.w[1] * s_sem + self.w[2] * s_beh
        label = int(prcs > self.tau)

        return PRCSResult(
            prcs=prcs,
            label=label,
            s_meta=s_meta,
            s_sem=s_sem,
            s_beh=s_beh,
            llm_reasoning=features.llm_reasoning,
            attack_type=features.llm_attack_type,
            risk_indicators=features.llm_risk_indicators,
        )


# ------------------------------------------------------------------
# LLM-Sentry: Main Pipeline Orchestrator
# ------------------------------------------------------------------

class LLMSentry:
    """
    Six-stage malicious package detection pipeline.

    Usage:
        sentry = LLMSentry()
        result = sentry.analyze("/path/to/package.tar.gz")
        print(f"PRCS={result.prcs:.3f}, label={result.label}")
    """

    def __init__(self,
                 openai_api_key: Optional[str] = None,
                 llm_model: str = "gpt-4o-2024-11-20",
                 prcs_weights: list[float] = PRCS_WEIGHTS,
                 threshold: float = PRCS_THRESHOLD):
        self.ingestion = PackageIngestionModule()
        self.meta_extractor = MetadataExtractor()
        self.code_embedder = CodeEmbeddingModule()
        self.llm_analyzer = LLMAnalyzer(api_key=openai_api_key, model=llm_model)
        self.behavioral = BehavioralAnalyzer()
        self.prcs_engine = PRCSEngine(weights=prcs_weights, threshold=threshold)

    def analyze(self, archive_path: str,
                extra_manifest: Optional[dict] = None) -> PRCSResult:
        """Run the full six-stage pipeline on a package archive."""

        # Stage 1: Ingestion
        vfs = self.ingestion.ingest(archive_path)
        manifest = self.ingestion.extract_manifest(vfs)
        if extra_manifest:
            manifest.update(extra_manifest)
        install_scripts = self.ingestion.extract_install_scripts(vfs)

        # Stage 2: Metadata
        meta_vec = self.meta_extractor.extract(manifest, vfs, install_scripts)

        # Stage 3: Code Embedding
        code_emb = self.code_embedder.embed(vfs)
        suspicious_nodes = self.code_embedder.extract_suspicious_ast_nodes(vfs)

        # Stage 4: LLM Analysis
        llm_output = self.llm_analyzer.analyze(
            manifest, install_scripts, suspicious_nodes
        )

        # Stage 5: Behavioral Analysis
        api_seq, b_dev = self.behavioral.analyze_static(install_scripts)

        # Assemble features
        features = PackageFeatures(
            name=manifest.get("name", "unknown"),
            version=manifest.get("version", "0.0.0"),
            ecosystem=manifest.get("_ecosystem", "npm"),
        )
        (features.name_dist, features.maint_age, features.version_count,
         features.dl_velocity, features.script_entropy, features.dep_count,
         features.repo_present, features.license_present,
         features.has_postinstall, features.file_count,
         features.obfusc_ratio, features.url_in_code, features.eval_count,
         features.b64_count, features.env_access, features.net_call_count,
         features.fs_write_count, features.pkg_size, features.desc_length,
         features.keyword_count, features.author_email_entropy,
         features.publish_date_days) = meta_vec.tolist()

        features.code_embedding = code_emb
        features.llm_confidence = llm_output["confidence"]
        features.llm_attack_type = llm_output["attack_type"]
        features.llm_risk_indicators = llm_output["risk_indicators"]
        features.llm_reasoning = llm_output["reasoning"]
        features.api_sequence = api_seq
        features.behavioral_deviation = b_dev

        # Stage 6: PRCS
        return self.prcs_engine.compute(features)


# ------------------------------------------------------------------
# CLI Entry Point
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="LLM-Sentry: Malicious Package Detector"
    )
    parser.add_argument("archive", help="Path to package archive (.tar.gz, .whl, .tgz)")
    parser.add_argument("--threshold", type=float, default=PRCS_THRESHOLD,
                        help="PRCS decision threshold (default: 0.55)")
    parser.add_argument("--llm-model", default="gpt-4o-2024-11-20",
                        help="OpenAI model name")
    args = parser.parse_args()

    sentry = LLMSentry(threshold=args.threshold, llm_model=args.llm_model)
    result = sentry.analyze(args.archive)

    print(json.dumps({
        "archive": args.archive,
        "prcs": round(result.prcs, 4),
        "label": "MALICIOUS" if result.label else "BENIGN",
        "s_meta": round(result.s_meta, 4),
        "s_sem": round(result.s_sem, 4),
        "s_beh": round(result.s_beh, 4),
        "attack_type": result.attack_type,
        "risk_indicators": result.risk_indicators,
        "llm_reasoning": result.llm_reasoning,
    }, indent=2))
