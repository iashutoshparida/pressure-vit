"""Run every experiment reported in the README and write results/results.json.

Usage:  python scripts/run_experiments.py --data <path>/experiment-i [--quick]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pvit import PressureViT, ViTConfig, attention_rollout, count_params  # noqa: E402
from pvit.baselines import SmallCNN, logistic_proba  # noqa: E402
from pvit.data import POSTURE_NAMES, load_cached  # noqa: E402
from pvit.metrics import summarise  # noqa: E402
from pvit.train import TrainConfig, fit, labels_for, loso, predict_proba  # noqa: E402

OUT = os.environ.get("PVIT_OUT", os.path.join(os.path.dirname(__file__), "..", "results"))
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")


def save(results):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


def log(msg):
    print(msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--quick", action="store_true", help="2 subjects, 2 epochs: smoke test only")
    ap.add_argument("--ablations", action="store_true", help="also run CNN baseline and ViT ablations")
    ap.add_argument("--only", nargs="*", default=None,
                    help="run only these steps and merge into an existing results.json")
    args = ap.parse_args()

    data = load_cached(args.data)
    subjects = [1, 2] if args.quick else None
    ep = 2 if args.quick else None
    results = {"dataset": {"frames": int(len(data.x)), "subjects": 13, "positions": 17},
               "params": {"vit": count_params(PressureViT(ViTConfig(num_classes=17))),
                          "cnn": count_params(SmallCNN(17))},
               "runs": {}}
    prev = os.path.join(OUT, "results.json")
    if args.only and os.path.exists(prev):
        results["runs"] = json.load(open(prev)).get("runs", {})

    def want(step):
        return args.only is None or step in args.only
    log(f"frames={len(data.x)}  vit_params={results['params']['vit']}  cnn_params={results['params']['cnn']}")

    def record(name, proba, tested, task):
        y = labels_for(task, data, np.arange(len(data.x)))
        r = summarise(name, proba[tested], y[tested], data.subject[tested])
        r["task"] = task
        results["runs"][name] = r
        save(results)
        log(f"== {name}: acc={r['accuracy']:.4f} CI={r['accuracy_ci95']} macroF1={r['macro_f1']:.4f} "
            f"worst={r['worst_subject_acc']:.3f} ECE={r['ece']:.3f}")
        np.save(os.path.join(OUT, f"proba_{name}.npy"), proba)

    plan = [
        ("vit_posture_loso", "posture", lambda: PressureViT(ViTConfig(num_classes=3)), dict(epochs=8)),
        ("vit_position_loso", "position", lambda: PressureViT(ViTConfig(num_classes=17)), dict(epochs=14)),
    ]
    if args.ablations:
        plan += [
            ("cnn_position_loso", "position", lambda: SmallCNN(17), dict(epochs=14)),
            ("vit_position_loso_no_posemb", "position",
             lambda: PressureViT(ViTConfig(num_classes=17, pos_embedding=False)), dict(epochs=14)),
            ("vit_position_loso_no_mirror", "position", lambda: PressureViT(ViTConfig(num_classes=17)),
             dict(epochs=14, mirror_aug=False)),
        ]
    # Longer schedule: at 14 epochs the ViT is still underfitting (87% accuracy on its own
    # training frames), so it is also run for 40 epochs. Chosen from training accuracy,
    # never from held-out accuracy.
    long_runs = [
        ("vit_position_loso_40ep", "position", lambda: PressureViT(ViTConfig(num_classes=17)), dict(epochs=40)),
    ]
    if args.only:
        plan = [p for p in plan + long_runs + [
            ("cnn_position_loso", "position", lambda: SmallCNN(17), dict(epochs=14)),
            ("vit_position_loso_no_posemb", "position",
             lambda: PressureViT(ViTConfig(num_classes=17, pos_embedding=False)), dict(epochs=14)),
            ("vit_position_loso_no_mirror", "position", lambda: PressureViT(ViTConfig(num_classes=17)),
             dict(epochs=14, mirror_aug=False)),
        ] if want(p[0])]
        seen = set()
        plan = [p for p in plan if not (p[0] in seen or seen.add(p[0]))]
    for name, task, make, kw in plan:
        if ep:
            kw = {**kw, "epochs": ep}
        cfg = TrainConfig(task=task, train_stride=4, **kw)
        log(f"\n### {name}  {cfg}")
        t0 = time.time()
        proba, tested = loso(data, make, cfg, subjects=subjects, log=log)
        record(name, proba, tested, task)
        results["runs"][name]["minutes"] = round((time.time() - t0) / 60, 1)
        save(results)

    # Logistic regression baseline (17 positions, LOSO)
    if want("logreg_position_loso"):
        run_logreg(data, subjects, record)
    if want("vit_position_random_frames"):
        run_random_split(data, ep or 14, record, "vit_position_random_frames")
    if want("vit_position_random_frames_40ep"):
        run_random_split(data, ep or 40, record, "vit_position_random_frames_40ep")
    if want("rollout"):
        run_rollout(data, ep)
    log("done")


def run_logreg(data, subjects, record):
    log("\n### logreg_position_loso")
    proba = np.zeros((len(data.x), 17), dtype=np.float32)
    tested = np.zeros(len(data.x), dtype=bool)
    for s in (subjects or range(1, 14)):
        tr = np.where((data.subject != s) & (data.frame % 4 == 0))[0]
        te = np.where(data.subject == s)[0]
        proba[te] = logistic_proba(data.x[tr], labels_for("position", data, tr), data.x[te], 17)
        tested[te] = True
    record("logreg_position_loso", proba, tested, "position")


def run_random_split(data, epochs, record, name):
    # Leakage demo: random frame-level split lets the same person's near-identical frames
    # sit on both sides of the split. Same model and budget as the matching LOSO run.
    log(f"\n### {name} (leaky split, {epochs} epochs)")
    rng = np.random.default_rng(0)
    is_test = rng.random(len(data.x)) < 0.2
    tr = np.where(~is_test & (data.frame % 4 == 0))[0]
    te = np.where(is_test)[0]
    cfg = TrainConfig(task="position", epochs=epochs, train_stride=1)
    model = fit(PressureViT(ViTConfig(num_classes=17)), data.x[tr], labels_for("position", data, tr), cfg)
    proba = np.zeros((len(data.x), 17), dtype=np.float32)
    proba[te] = predict_proba(model, data.x[te])
    record(name, proba, is_test, "position")


def run_rollout(data, ep):
    # Attention rollout: train the 3-posture ViT on 12 people, look at held-out S5.
    log("\n### attention rollout (held-out S5)")
    tr = np.where((data.subject != 5) & (data.frame % 4 == 0))[0]
    model = fit(PressureViT(ViTConfig(num_classes=3)), data.x[tr], labels_for("posture", data, tr),
                TrainConfig(task="posture", epochs=ep or 8))
    model.eval()
    maps = {}
    with torch.no_grad():
        for c, cname in enumerate(POSTURE_NAMES):
            idx = np.where((data.subject == 5) & (data.posture == c))[0]
            x = torch.from_numpy(data.x[idx]).unsqueeze(1)
            _, attns = model(x, return_attn=True)
            maps[cname] = {"rollout": attention_rollout(attns).mean(0).view(8, 4).numpy().tolist(),
                           "mean_frame": data.x[idx].mean(0).tolist()}
    with open(os.path.join(OUT, "attention_rollout_S5.json"), "w") as f:
        json.dump(maps, f)


if __name__ == "__main__":
    main()
