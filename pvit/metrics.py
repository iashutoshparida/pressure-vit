"""Metrics with uncertainty: subject-level bootstrap and calibration."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def per_subject_accuracy(y, pred, subject):
    subs = sorted(set(subject.tolist()))
    return {s: float((pred[subject == s] == y[subject == s]).mean()) for s in subs}


def subject_bootstrap_ci(y, pred, subject, metric="acc", n_boot=2000, seed=0):
    """Resample *people*, not frames: frames from one person are not independent."""
    rng = np.random.default_rng(seed)
    subs = np.array(sorted(set(subject.tolist())))
    idx_of = {s: np.where(subject == s)[0] for s in subs}
    stats = []
    for _ in range(n_boot):
        pick = rng.choice(subs, size=len(subs), replace=True)
        idx = np.concatenate([idx_of[s] for s in pick])
        if metric == "acc":
            stats.append((pred[idx] == y[idx]).mean())
        else:
            stats.append(f1_score(y[idx], pred[idx], average="macro"))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def expected_calibration_error(proba, y, n_bins=15):
    conf = proba.max(1)
    correct = (proba.argmax(1) == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def summarise(name, proba, y, subject):
    pred = proba.argmax(1)
    per = per_subject_accuracy(y, pred, subject)
    acc = float((pred == y).mean())
    f1 = float(f1_score(y, pred, average="macro"))
    return {
        "model": name,
        "accuracy": acc,
        "accuracy_ci95": subject_bootstrap_ci(y, pred, subject, "acc"),
        "macro_f1": f1,
        "macro_f1_ci95": subject_bootstrap_ci(y, pred, subject, "f1"),
        "worst_subject_acc": float(min(per.values())),
        "subject_acc_std": float(np.std(list(per.values()))),
        "ece": expected_calibration_error(proba, y),
        "per_subject_acc": per,
    }
