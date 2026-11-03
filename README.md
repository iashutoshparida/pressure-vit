# pressure-vit

A vision transformer written from scratch in PyTorch that reads a bed's pressure map and tells how a person is lying.
It is tested only on people it has never seen.

One frame is a 64 x 32 grid of pressure sensors (2,048 readings, sampled once a second).
The model predicts either **3 postures** (supine, right side, left side) or **all 17 recorded positions**.

## What is written by hand

- **Multi-head self-attention** (`pvit/attention.py`): queries, keys and values, scaling by 1/sqrt(d_k), masking, softmax and merging heads.
  Tests check it against PyTorch's `scaled_dot_product_attention` and `nn.MultiheadAttention` to 1e-6.
- **Patch embedding**: the frame is cut into 8 x 8 tiles (32 patches) with reshape and permute, then projected to 64-d tokens.
- **ViT**: a [CLS] token, learned positional embedding and 4 pre-LayerNorm blocks. About 0.14M parameters.
- **Attention rollout** (Abnar & Zuidema, 2020), to see which body regions the model uses.

Only `nn.Linear`, `nn.LayerNorm` and the optimiser come from PyTorch.

## How it is evaluated

- **Leave one subject out.** For each of the 13 people, train on the other 12 and test on that person. Every headline number comes from unseen people.
- **95% intervals by bootstrapping people, not frames**, because frames from one person are not independent.
- Also reported: macro-F1, worst-person accuracy and expected calibration error (ECE).
- **Leakage check.** The same model is also trained on a random frame-level split.
  Near-identical frames from one recording then fall on both sides of the split, and the gap from the subject-level number shows how much a leaky split inflates results.
- **Baselines and ablations**: logistic regression on pooled pixels, a CNN with a similar parameter budget, a ViT without positional embedding, and a ViT without mirror augmentation.

Hyperparameters are fixed in advance and never tuned on a held-out person.

## Data handling

- Readings above the sensor's 1,000 limit (0.56% of all readings) are clipped as hot pixels.
- Each frame is divided by its total load, so the model learns body shape rather than body weight.
- The first 3 frames of each recording are dropped while the person settles.
- Training uses every 4th frame, since neighbouring 1 Hz frames are nearly identical. Testing uses every frame.
- Augmentation: shifts of up to 2 sensors, +/-15% gain jitter, and a left-right mirror that **swaps the label** too (a right-side frame mirrored becomes a left-side frame).

The reasoning behind each choice is in [`docs/DESIGN_NOTES.md`](docs/DESIGN_NOTES.md).

## Run it

```bash
pip install -r requirements.txt
pytest -q                                   # attention parity, masking, patchify, shapes

# Download Experiment I from PhysioNet (link below), then:
python scripts/run_experiments.py --data <path>/experiment-i --ablations
python scripts/make_figures.py <path>/experiment-i
```

Everything runs on a CPU. `--quick` runs a 2-person, 2-epoch smoke test.

## Layout

```
pvit/attention.py    multi-head self-attention
pvit/model.py        patch embedding, ViT, attention rollout
pvit/data.py         loading, preprocessing, position-to-posture and mirror maps
pvit/train.py        augmentation, training loop, leave-one-subject-out
pvit/metrics.py      subject bootstrap CIs, macro-F1, ECE
pvit/baselines.py    CNN and logistic regression
scripts/             experiments and figures
tests/               unit tests
```

## Data and licence

The data is **not** included in this repository. It is "A Pressure Map Dataset for In-bed Posture Classification" (v1.0.0) on PhysioNet,
released under the [Open Data Commons Attribution License v1.0](https://physionet.org/content/pmd/view-license/1.0.0/).
This project uses Experiment I: 13 people, 17 positions, recorded on a Vista Medical FSA SoftFlex 2048 mat.

- Dataset: https://physionet.org/content/pmd/1.0.0/ (DOI [10.13026/C2W09K](https://doi.org/10.13026/C2W09K))
- M. B. Pouyan, J. Birjandtalab, M. Heydarzadeh, M. Nourani and S. Ostadabbas, "A pressure map dataset for posture and subject analytics,"
  *2017 IEEE EMBS International Conference on Biomedical & Health Informatics (BHI)*, 2017. DOI [10.1109/BHI.2017.7897206](https://doi.org/10.1109/BHI.2017.7897206)
- PhysioNet: T. Pollard et al., "PhysioNet as a global platform for biomedical research," *Nature Health*, 1(8), 792–795, 2026.
  DOI [10.1038/s44360-026-00096-z](https://doi.org/10.1038/s44360-026-00096-z)
