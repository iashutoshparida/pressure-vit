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

    # 1. Accuracy per model on 17 positions, with subject-bootstrap 95% CI.
    # Blue: tested on people the model never saw. Orange: leaky random-frame split.
    surface, ink, muted, axis = "#fcfcfb", "#0b0b0b", "#898781", "#c3c2b7"
    blue, orange = "#2a78d6", "#eb6834"
    labels = {"logreg_position_loso": "Logistic regression (14 ep.)",
              "vit_position_loso_no_posemb": "ViT, no positional emb. (14 ep.)",
              "vit_position_loso_no_mirror": "ViT, no mirror aug. (14 ep.)",
              "vit_position_loso": "ViT (14 ep.)",
              "cnn_position_loso": "CNN (14 ep.)",
              "vit_position_loso_40ep": "ViT (40 ep.)",
              "vit_position_random_frames": "ViT, leaky split (14 ep.)",
              "vit_position_random_frames_40ep": "ViT, leaky split (40 ep.)"}
    order = [k for k in labels if k in runs]
    acc = np.array([runs[k]["accuracy"] * 100 for k in order])
    lo = np.array([runs[k]["accuracy_ci95"][0] * 100 for k in order])
    hi = np.array([runs[k]["accuracy_ci95"][1] * 100 for k in order])
    leaky = ["random_frames" in k for k in order]
    fig, ax = plt.subplots(figsize=(7.6, 0.42 * len(order) + 1.2), facecolor=surface)
    ax.set_facecolor(surface)
    y = np.arange(len(order))[::-1]
    ax.barh(y, acc, height=0.62, color=[orange if lk else blue for lk in leaky], edgecolor=surface, linewidth=2)
    ax.errorbar(acc, y, xerr=[acc - lo, hi - acc], fmt="none", ecolor=muted, elinewidth=1.2, capsize=3)
    for yi, a, h in zip(y, acc, hi):
        ax.text(h + 1.2, yi, f"{a:.1f}%", va="center", fontsize=9, color=ink)
    ax.set_yticks(y, [labels[k] for k in order], fontsize=9, color=ink)
    ax.set_xlim(0, 110)
    ax.set_xticks(range(0, 101, 20))
    ax.tick_params(axis="x", colors=muted, labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Accuracy on 17 positions (%), with 95% CI over people", fontsize=9, color=muted)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(axis)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=blue, label="Tested on unseen people"),
                       Patch(color=orange, label="Leaky random-frame split")],
              loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=2, frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "accuracy_by_model.png"), dpi=160, facecolor=surface)
    plt.close()

    # 2. Attention rollout on a held-out person
    # Top row: mean pressure. Bottom row: share of [CLS] attention per 8 x 8 patch.
    maps = json.load(open(os.path.join(RES, "attention_rollout_S5.json")))
    vmax = max(np.max(m["rollout"]) for m in maps.values())
    fig, axes = plt.subplots(2, 3, figsize=(6.4, 7.2), facecolor=surface)
    for j, (name, m) in enumerate(maps.items()):
        axes[0, j].imshow(np.array(m["mean_frame"]), cmap="Greys")
        axes[0, j].set_title(name, fontsize=10, color=ink)
        im = axes[1, j].imshow(np.array(m["rollout"]), cmap="Blues", vmin=0, vmax=vmax)
        for a in axes[:, j]:
            a.set_xticks([])
            a.set_yticks([])
            for s in a.spines.values():
                s.set_color(axis)
    axes[0, 0].set_ylabel("mean pressure", fontsize=9, color=muted)
    axes[1, 0].set_ylabel("attention to [CLS]", fontsize=9, color=muted)
    cb = fig.colorbar(im, ax=axes[1, :], fraction=0.03, pad=0.02)
    cb.ax.tick_params(labelsize=8, colors=muted)
    cb.outline.set_visible(False)
    fig.suptitle("Attention rollout on held-out subject S5 (3-posture ViT)", fontsize=10, color=ink)
    plt.savefig(os.path.join(FIG, "attention_rollout.png"), dpi=160, facecolor=surface, bbox_inches="tight")
    plt.close()

    # 3. Confusion matrix, 17 positions, ViT LOSO (the longest-trained run available)
    best = next((k for k in ["vit_position_loso_40ep", "vit_position_loso"] if k in runs), None)
    if best:
        data = load_cached(data_root)
        proba = np.load(os.path.join(RES, f"proba_{best}.npy"))
        y = data.position - 1
        cm = confusion_matrix(y, proba.argmax(1), normalize="true")
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(17), range(1, 18), fontsize=7)
        ax.set_yticks(range(17), range(1, 18), fontsize=7)
        ax.set_xlabel("predicted position")
        ax.set_ylabel("true position")
        plt.colorbar(im, fraction=0.046)
        ep = "40" if best.endswith("40ep") else "14"
        ax.set_title(f"ViT ({ep} ep.), leave-one-subject-out, row-normalised", fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG, "confusion_17.png"), dpi=160)
        plt.close()
    print("figures written to", FIG)


if __name__ == "__main__":
    main(sys.argv[1])
