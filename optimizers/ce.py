from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

def save_dist(path: Path, P_map: Any) -> None:
    path.write_text(json.dumps(P_map, indent=2) + "\n")

def load_dist(path: Path) -> Any:
    return json.loads(path.read_text())

BASES_RNA = ("A", "C", "G", "U")
BASES_DNA = ("A", "C", "G", "T")


@dataclass
class CEConfig:
    """
    Cross-Entropy (CE) optimizer config for factorized categorical distribution.

    Notes:
    - batch_size: number of sequences sampled per iteration
    - elite_frac: fraction of best sequences selected as elites
    - smoothing_alpha: exponential smoothing for distribution update
    - temperature: softens sampling from logits (higher -> more uniform)
    - min_prob: lower bound per-base per-position to avoid early collapse
    """
    length: int = 30
    alphabet: str = "RNA"  # "RNA" or "DNA"
    batch_size: int = 64
    elite_frac: float = 0.1
    smoothing_alpha: float = 0.4
    temperature: float = 1.0
    min_prob: float = 0.02
    seed: int = 123

    def validate(self) -> None:
        if self.length <= 0:
            raise ValueError(f"length must be >0, got {self.length}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be >0, got {self.batch_size}")
        if not (0.0 < self.elite_frac <= 1.0):
            raise ValueError(f"elite_frac must be in (0,1], got {self.elite_frac}")
        if not (0.0 <= self.smoothing_alpha <= 1.0):
            raise ValueError(f"smoothing_alpha must be in [0,1], got {self.smoothing_alpha}")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be >0, got {self.temperature}")
        if not (0.0 <= self.min_prob < 0.25):
            # 4-letter alphabet; min_prob>=0.25 would force uniform
            raise ValueError(f"min_prob must be in [0,0.25), got {self.min_prob}")
        if self.alphabet.upper() not in ("RNA", "DNA"):
            raise ValueError(f"alphabet must be 'RNA' or 'DNA', got {self.alphabet}")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CEConfig":
        """
        Build CEConfig from a dict, with backward-compatible aliases.

        Supported aliases (common in configs):
          - batch -> batch_size
          - alpha -> smoothing_alpha
        """
        dd = dict(d)

        # aliases
        if "batch" in dd and "batch_size" not in dd:
            dd["batch_size"] = dd.pop("batch")
        if "alpha" in dd and "smoothing_alpha" not in dd:
            dd["smoothing_alpha"] = dd.pop("alpha")

        cfg = cls(**dd)
        cfg.validate()
        return cfg


def _alphabet(cfg: CEConfig) -> Tuple[str, ...]:
    return BASES_DNA if cfg.alphabet.upper() == "DNA" else BASES_RNA


def empirical_freqs(
    seqs: Sequence[str],
    bases: Sequence[str],
    length: int,
    eps: float = 0.0,
) -> np.ndarray:
    """
    Compute empirical per-position base frequencies from sequences.
    Returns array shape [L, K].

    eps: optional pseudocount for smoothing (Laplace-like).
    """
    K = len(bases)
    base_to_idx = {b: i for i, b in enumerate(bases)}
    counts = np.full((length, K), float(eps), dtype=np.float64)

    for s in seqs:
        if len(s) != length:
            raise ValueError(f"Sequence length mismatch: expected {length}, got {len(s)}")
        for i, ch in enumerate(s):
            try:
                counts[i, base_to_idx[ch]] += 1.0
            except KeyError:
                raise ValueError(f"Invalid base '{ch}' for alphabet {bases}")

    freqs = counts / counts.sum(axis=1, keepdims=True)
    return freqs


def shannon_entropy(p: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """
    Shannon entropy H(p) = -sum p log p.
    """
    p = np.clip(p, eps, 1.0)
    return -(p * np.log(p)).sum(axis=axis)


def kl_divergence(p: np.ndarray, q: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """
    KL(p || q) = sum p log(p/q)
    """
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return (p * (np.log(p) - np.log(q))).sum(axis=axis)


def select_elites(
    seqs: Sequence[str],
    scores: Sequence[float],
    elite_frac: float,
    minimize: bool = True,
) -> Tuple[List[str], List[float], float]:
    """
    Select elite sequences based on scores.

    minimize=True  -> smaller score is better (e.g., Vina affinity)
    minimize=False -> larger score is better

    Returns: (elite_seqs, elite_scores, threshold_score)
    """
    if len(seqs) != len(scores):
        raise ValueError("seqs and scores must have same length")

    n = len(seqs)
    if n == 0:
        return [], [], float("nan")

    elite_n = max(1, int(np.ceil(elite_frac * n)))
    idx = np.argsort(scores)
    if not minimize:
        idx = idx[::-1]
    elite_idx = idx[:elite_n]

    elite_seqs = [seqs[i] for i in elite_idx]
    elite_scores = [float(scores[i]) for i in elite_idx]
    thr = elite_scores[-1]
    return elite_seqs, elite_scores, thr


class SequenceDistribution:
    """
    Position-independent categorical distribution over bases.
    Represented by logits L x K. probs = softmax(logits / temperature).
    """

    def __init__(self, cfg: CEConfig):
        cfg.validate()
        self.cfg = cfg
        self.bases = _alphabet(cfg)
        self.L = cfg.length
        self.K = len(self.bases)
        # uniform at start: logits all zeros
        self.logits = np.zeros((self.L, self.K), dtype=np.float64)

    def probs(self) -> np.ndarray:
        """
        Return probability matrix [L, K] after temperature + min_prob constraint.
        """
        z = self.logits / max(self.cfg.temperature, 1e-8)
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(axis=1, keepdims=True)

        if self.cfg.min_prob > 0:
            p = np.clip(p, self.cfg.min_prob, 1.0)
            p = p / p.sum(axis=1, keepdims=True)
        return p

    def sample(self, n: int, rng: np.random.Generator) -> List[str]:
        """
        Sample n sequences from current distribution.
        """
        if n <= 0:
            return []
        p = self.probs()
        seqs: List[str] = []
        for _ in range(n):
            idx = [rng.choice(self.K, p=p[i]) for i in range(self.L)]
            seqs.append("".join(self.bases[j] for j in idx))
        return seqs

    def update_from_elites(self, elite_seqs: Sequence[str]) -> None:
        """
        Update logits from elite sequences using:
          logits <- (1-a)*logits + a*log(freqs)
        with min_prob enforced on freqs.
        """
        if not elite_seqs:
            return

        freqs = empirical_freqs(elite_seqs, self.bases, self.L, eps=0.0)

        if self.cfg.min_prob > 0:
            freqs = np.clip(freqs, self.cfg.min_prob, 1.0)
            freqs = freqs / freqs.sum(axis=1, keepdims=True)

        new_logits = np.log(freqs + 1e-12)
        a = float(self.cfg.smoothing_alpha)
        self.logits = (1.0 - a) * self.logits + a * new_logits

    def position_entropy(self) -> np.ndarray:
        """
        Entropy per position: shape [L]
        """
        return shannon_entropy(self.probs(), axis=1)

    def mean_entropy(self) -> float:
        return float(self.position_entropy().mean())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cfg": asdict(self.cfg),
            "bases": list(self.bases),
            "logits": self.logits.tolist(),
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_file(cls, path: Path) -> "SequenceDistribution":
        d = json.loads(Path(path).read_text())
        cfg = CEConfig.from_dict(d["cfg"])
        obj = cls(cfg)
        obj.logits = np.array(d["logits"], dtype=np.float64)
        return obj


class CEOptimizer:
    """
    Lightweight CE driver to manage RNG + distribution.
    Your ce_design.py can use this, or directly use SequenceDistribution.

    Typical loop:
      opt = CEOptimizer(cfg)
      seqs = opt.sample_batch()
      scores = evaluate(seqs)
      elites, _, _ = select_elites(...)
      opt.update(elites)
    """

    def __init__(self, cfg: CEConfig):
        cfg.validate()
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.dist = SequenceDistribution(cfg)

    def sample_batch(self, n: Optional[int] = None) -> List[str]:
        return self.dist.sample(int(n or self.cfg.batch_size), self.rng)

    def update(self, elite_seqs: Sequence[str]) -> None:
        self.dist.update_from_elites(elite_seqs)

    def save_state(self, path: Path) -> None:
        """
        Save distribution only (enough for resume).
        """
        self.dist.save(path)

    def load_state(self, path: Path) -> None:
        """
        Load distribution and overwrite current.
        """
        loaded = SequenceDistribution.from_file(path)
        self.cfg = loaded.cfg
        self.dist = loaded
        self.rng = np.random.default_rng(self.cfg.seed)
