"""Figures for the README from results/ (run run_experiments.py first)."""
from __future__ import annotations

import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pvit.data import load_cached  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")


def main(data_root):
    os.makedirs(FIG, exist_ok=True)
    res = json.load(open(os.path.join(RES, "results.json")))
    runs = res["runs"]

    # 1. Held-out accuracy per model (17 positions), with subject-bootstrap 95% CI
    order = [k for k in ["logreg_position_loso", "cnn_position_loso", "vit_position_loso_no_posemb",
                         "vit_position_loso_no_mirror", "vit_position_loso", "vit_position_random_frames"]
             if k in runs]
    labels = {"logreg_position_loso": "Logistic reg.", "cnn_position_loso": "CNN",
              "vit_position_loso_no_posemb": "ViT, no pos-emb", "vit_position_loso_no_mirror": "ViT, no mirror aug",
              "vit_position_loso": "ViT (ours)", "vit_position_random_frames": "ViT, leaky frame split"}
    acc = [runs[k]["accuracy"] * 100 for k in order]
    lo = [runs[k]["accuracy_ci95"][0] * 100 for k in order]
    hi = [runs[k]["accuracy_ci95"][1] * 100 for k in order]
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    colors = ["#9aa5b1"] * len(order)
    for i, k in enumerate(order):
        if k == "vit_position_loso":
            colors[i] = "#2c4cb0"
        if k == "vit_position_random_frames":
            colors[i] = "#c0392b"
    ax.barh([labels[k] for k in order], acc, color=colors,
            xerr=[np.subtract(acc, lo), np.subtract(hi, acc)], capsize=3)
    for i, a in enumerate(acc):
        ax.text(a + 1, i, f"{a:.1f}%", va="center", fontsize=9)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Accuracy on 17 positions (%)  - all bars except red test on unseen people")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "accuracy_by_model.png"), dpi=160)
    plt.close()

    # 2. Attention rollout on a held-out person
    maps = json.load(open(os.path.join(RES, "attention_rollout_S5.json")))
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 4.6))
    for ax, (name, m) in zip(axes, maps.items()):
        frame = np.array(m["mean_frame"])
        roll = np.kron(np.array(m["rollout"]), np.ones((8, 8)))
        ax.imshow(frame, cmap="gray_r")
        ax.imshow(roll, cmap="magma", alpha=0.55)
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle("Where [CLS] looks: attention rollout on held-out subject S5", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "attention_rollout.png"), dpi=160)
    plt.close()

    # 3. Confusion matrix, 17 positions, ViT LOSO
    if "vit_position_loso" in runs:
        data = load_cached(data_root)
        proba = np.load(os.path.join(RES, "proba_vit_position_loso.npy"))
        y = data.position - 1
        cm = confusion_matrix(y, proba.argmax(1), normalize="true")
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(17), range(1, 18), fontsize=7)
        ax.set_yticks(range(17), range(1, 18), fontsize=7)
        ax.set_xlabel("predicted position")
        ax.set_ylabel("true position")
        plt.colorbar(im, fraction=0.046)
        ax.set_title("ViT, leave-one-subject-out (row-normalised)", fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG, "confusion_17.png"), dpi=160)
        plt.close()
    print("figures written to", FIG)


if __name__ == "__main__":
    main(sys.argv[1])
