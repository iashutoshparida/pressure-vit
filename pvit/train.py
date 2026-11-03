"""Training and leave-one-subject-out (LOSO) evaluation."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .data import MIRROR_POSITION, MIRROR_POSTURE


@dataclass
class TrainConfig:
    task: str = "posture"            # "posture" (3 classes) or "position" (17 classes)
    epochs: int = 12
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    train_stride: int = 2            # consecutive 1 Hz frames are near-duplicates
    mirror_aug: bool = True          # left-right mirror with label swap
    shift_px: int = 2                # random translation on the mat
    gain_jitter: float = 0.15        # sensor gain / calibration drift
    seed: int = 0
    threads: int = 2
    extra: dict = field(default_factory=dict)


def labels_for(task: str, data, idx: np.ndarray) -> np.ndarray:
    return (data.posture[idx] if task == "posture" else data.position[idx] - 1).astype(np.int64)


def _mirror_labels(y: torch.Tensor, task: str) -> torch.Tensor:
    if task == "posture":
        lut = torch.tensor([MIRROR_POSTURE[i] for i in range(3)])
    else:
        lut = torch.tensor([MIRROR_POSITION[i + 1] - 1 for i in range(17)])
    return lut[y]


def augment(x: torch.Tensor, y: torch.Tensor, cfg: TrainConfig, g: torch.Generator):
    b = x.shape[0]
    if cfg.mirror_aug:
        flip = torch.rand(b, generator=g) < 0.5
        x = torch.where(flip.view(-1, 1, 1, 1), x.flip(-1), x)
        y = torch.where(flip, _mirror_labels(y, cfg.task), y)
    if cfg.shift_px:
        dy, dx = (int(v) for v in torch.randint(-cfg.shift_px, cfg.shift_px + 1, (2,), generator=g))
        x = torch.roll(x, shifts=(dy, dx), dims=(-2, -1))
    if cfg.gain_jitter:
        gain = 1 + cfg.gain_jitter * (2 * torch.rand(b, 1, 1, 1, generator=g) - 1)
        x = x * gain
    return x, y


def fit(model: nn.Module, x: np.ndarray, y: np.ndarray, cfg: TrainConfig) -> nn.Module:
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.threads)
    g = torch.Generator().manual_seed(cfg.seed)
    xt = torch.from_numpy(x).unsqueeze(1)
    yt = torch.from_numpy(y)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * int(np.ceil(len(xt) / cfg.batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=steps, pct_start=0.15)
    model.train()
    for _ in range(cfg.epochs):
        order = torch.randperm(len(xt), generator=g)
        for i in range(0, len(xt), cfg.batch_size):
            bi = order[i:i + cfg.batch_size]
            xb, yb = augment(xt[bi], yt[bi], cfg, g)
            loss = F.cross_entropy(model(xb), yb, label_smoothing=cfg.label_smoothing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
    return model


@torch.no_grad()
def predict_proba(model: nn.Module, x: np.ndarray, batch: int = 512) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(x), batch):
        out.append(torch.softmax(model(torch.from_numpy(x[i:i + batch]).unsqueeze(1)), -1))
    return torch.cat(out).numpy()


def loso(data, make_model, cfg: TrainConfig, subjects=None, log=print):
    """Train on 12 people, test on the 13th, for every person. Returns pooled predictions."""
    subjects = sorted(set(data.subject.tolist())) if subjects is None else subjects
    n_cls = 3 if cfg.task == "posture" else 17
    proba = np.zeros((len(data.subject), n_cls), dtype=np.float32)
    tested = np.zeros(len(data.subject), dtype=bool)
    for s in subjects:
        t0 = time.time()
        tr = np.where(data.subject != s)[0]
        tr = tr[data.frame[tr] % cfg.train_stride == 0]
        te = np.where(data.subject == s)[0]
        model = make_model()
        fit(model, data.x[tr], labels_for(cfg.task, data, tr), cfg)
        proba[te] = predict_proba(model, data.x[te])
        tested[te] = True
        acc = (proba[te].argmax(1) == labels_for(cfg.task, data, te)).mean()
        log(f"  held-out S{s:<2d} acc={acc:.3f}  ({time.time() - t0:.0f}s)")
    return proba, tested
