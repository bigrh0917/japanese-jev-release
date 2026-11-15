"""Metrics in pure numpy, no torch dependency, so the tests can exercise them directly.

L  [N, Kmax] letter logits; only the first n of each row are valid, the rest are filled with -inf
y  [N]       index of the correct option
P  [N, Kmax] probabilities softmax(L / T)
"""

from __future__ import annotations

import numpy as np


def softmax(L: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = np.asarray(L, dtype=np.float64) / float(T)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)                      # -inf → 0
    return e / e.sum(1, keepdims=True)


def nll(T: float, L: np.ndarray, y: np.ndarray) -> float:
    """Negative log-likelihood, the objective minimized in calibration. T does not change the argmax, only the confidence."""
    z = np.asarray(L, dtype=np.float64) / float(T)
    z = z - z.max(1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(1, keepdims=True))
    return float(-logp[np.arange(len(y)), y].mean())


def accuracy(P: np.ndarray, y: np.ndarray) -> float:
    return float((P.argmax(1) == y).mean())


def within1(P: np.ndarray, y: np.ndarray) -> float:
    """Share of items whose argmax is within 1 level of the correct one. Only meaningful for score (an ordered scale): being one level off on a 5-level scale should not count as fully wrong."""
    return float((np.abs(P.argmax(1) - y) <= 1).mean())


def ece(P: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error: bin by max probability, take |mean confidence - accuracy| per bin, weight by bin size. Lower is better."""
    conf, pred = P.max(1), P.argmax(1)
    hit = (pred == y).astype(np.float64)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return float(total)


def brier(P: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error between the probability vector and the one-hot answer (over valid options only)."""
    onehot = np.zeros_like(P)
    onehot[np.arange(len(y)), y] = 1.0
    return float(((P - onehot) ** 2).sum(1).mean())    # at invalid positions both P and onehot are 0, so they do not contribute


def automation_rate(P: np.ndarray, y: np.ndarray, error_budget: float) -> float:
    """Share of items that can be handled automatically within an error budget: accept by descending confidence until the error rate of the accepted part exceeds the budget.

    Temperature scaling improves ECE but not necessarily this, so it is reported separately.
    """
    conf = P.max(1)
    hit = (P.argmax(1) == y)
    order = np.argsort(-conf)
    wrong = np.cumsum(~hit[order])
    k = np.arange(1, len(y) + 1)
    ok = wrong / k <= error_budget
    if not ok.any():
        return 0.0
    return float(k[ok].max() / len(y))


def reliability_table(P: np.ndarray, y: np.ndarray, bins: int = 15) -> list[dict]:
    """Data for a reliability diagram: mean confidence, accuracy and count per bin."""
    conf, pred = P.max(1), P.argmax(1)
    hit = (pred == y).astype(np.float64)
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            out.append({"lo": round(float(lo), 3), "hi": round(float(hi), 3), "count": int(m.sum()),
                        "confidence": round(float(conf[m].mean()), 4), "accuracy": round(float(hit[m].mean()), 4)})
    return out


def summary(L: np.ndarray, y: np.ndarray, T: float = 1.0, bins: int = 15) -> dict:
    P = softmax(L, T)
    return {
        "n": int(len(y)),
        "T": round(float(T), 4),
        "accuracy": round(accuracy(P, y), 4),
        "within1": round(within1(P, y), 4),
        "ece": round(ece(P, y, bins), 4),
        "brier": round(brier(P, y), 4),
        "nll": round(nll(T, L, y), 4),
        "automation_2pct": round(automation_rate(P, y, 0.02), 4),
        "automation_5pct": round(automation_rate(P, y, 0.05), 4),
    }
