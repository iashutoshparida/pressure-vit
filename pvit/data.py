"""Loading and preprocessing for the PhysioNet in-bed pressure-map dataset (Experiment I).

Pouyan et al., "A pressure map dataset for posture and subject analytics", IEEE BHI 2017.
13 subjects x 17 positions, ~2 min at 1 Hz each, Vista FSA SoftFlex 2048 mat (64 x 32 sensors).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

# From experiment-info.docx (posture table). Index -> coarse posture.
SUPINE, RIGHT, LEFT = 0, 1, 2
POSTURE_OF_POSITION = {
    1: SUPINE, 2: RIGHT, 3: LEFT, 4: RIGHT, 5: RIGHT, 6: LEFT, 7: LEFT,
    8: SUPINE, 9: SUPINE, 10: SUPINE, 11: SUPINE, 12: SUPINE,
    13: RIGHT, 14: LEFT, 15: SUPINE, 16: SUPINE, 17: SUPINE,
}
POSTURE_NAMES = ["supine", "right", "left"]

# Mirroring the mat left-right turns a right-side posture into its left-side twin.
MIRROR_POSITION = {1: 1, 2: 3, 3: 2, 4: 6, 6: 4, 5: 7, 7: 5, 8: 8, 9: 9, 10: 10,
                   11: 12, 12: 11, 13: 14, 14: 13, 15: 15, 16: 16, 17: 17}
MIRROR_POSTURE = {SUPINE: SUPINE, RIGHT: LEFT, LEFT: RIGHT}

H, W = 64, 32
SENSOR_MAX = 1000.0      # documented range; values above are isolated hot-pixel spikes
SETTLE_FRAMES = 3        # first seconds of each recording while the subject settles


@dataclass
class PressureData:
    x: np.ndarray          # (N, 64, 32) float32, preprocessed
    position: np.ndarray   # (N,) 1..17
    posture: np.ndarray    # (N,) 0..2
    subject: np.ndarray    # (N,) 1..13
    frame: np.ndarray      # (N,) index within its recording


def preprocess(raw: np.ndarray) -> np.ndarray:
    """Clip hot-pixel spikes, then normalise each frame by its total load.

    Dividing by total pressure removes most of the body-weight difference between
    people (48 kg vs 100 kg loads the mat very differently), which is the main
    nuisance variable when a model must generalise to an unseen person.
    """
    x = np.clip(raw, 0.0, SENSOR_MAX)
    total = x.reshape(len(x), -1).sum(axis=1).reshape(-1, 1, 1) + 1e-6
    return (x / total * (H * W)).astype(np.float32)


def load_experiment_i(root: str, subjects=range(1, 14)) -> PressureData:
    xs, pos, pst, sub, frm = [], [], [], [], []
    for s in subjects:
        for p in range(1, 18):
            raw = np.loadtxt(os.path.join(root, f"S{s}", f"{p}.txt"), dtype=np.float32)
            raw = raw[SETTLE_FRAMES:].reshape(-1, H, W)
            n = len(raw)
            xs.append(preprocess(raw))
            pos.append(np.full(n, p)); pst.append(np.full(n, POSTURE_OF_POSITION[p]))
            sub.append(np.full(n, s)); frm.append(np.arange(n))
    return PressureData(np.concatenate(xs), np.concatenate(pos), np.concatenate(pst),
                        np.concatenate(sub), np.concatenate(frm))


def cache_path(root: str) -> str:
    return os.path.join(root, "..", "pvit_cache_exp_i.npz")


def load_cached(root: str) -> PressureData:
    path = cache_path(root)
    if not os.path.exists(path):
        d = load_experiment_i(root)
        np.savez_compressed(path, x=d.x, position=d.position, posture=d.posture,
                            subject=d.subject, frame=d.frame)
    z = np.load(path)
    return PressureData(z["x"], z["position"], z["posture"], z["subject"], z["frame"])
