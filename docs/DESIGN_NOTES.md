# Design notes: why each choice was made

Read this before an interview. Every line on the resume should be explainable from here.

## The problem

A smart bed only sees pressure. To respond to a person, it first has to know how they are lying.
The model gets one 64 x 32 pressure frame (2,048 sensors, sampled once a second) and predicts either
3 postures (supine, right side, left side) or all 17 recorded positions (wedges, fetal, raised head, and so on).

The hard part is not the classifier. It is **generalising to a person the model has never seen**:
bodies differ in weight, height and where they press on the mat.

## Data handling

- **Hot-pixel clipping.** The mat is specified for readings of 0 to 1,000, but 0.56% of readings go up to
  4,095. These are isolated spikes, so they are clipped to 1,000.
- **Divide each frame by its total load.** A 100 kg person presses harder everywhere than a 63 kg person.
  Normalising by the frame's total removes most of that difference, so the model learns shape, not weight.
- **Drop the first 3 s of each recording.** The person is still settling.
- **Train on every 4th frame.** At 1 Hz, neighbouring frames are almost identical, so extra frames add cost, not information.
  Every frame is still used for testing.

## Evaluation: leave one person out

For each of the 13 people, train on the other 12 and test on that one person. Every reported number
comes from people the model never saw.

Why it matters: if frames are split at random, near-identical frames from the same two-minute recording
land in both training and test data. The model can then memorise the person instead of learning the posture.
This is the same leakage problem as patient-level splits in sleep-apnea ECG studies.
`vit_position_random_frames` runs exactly that leaky split, so the gap is measured, not assumed.

**Measured:** at 40 epochs the leaky split scores 98.3% against 90.7% on unseen people, a 7.6-point inflation,
even though the leaky split trains on fewer frames (3,926 against about 4,540). At 14 epochs there was no gap
(84.7% against 85.2%), because the model was still underfitting: a model that cannot yet memorise its training
frames gains nothing from seeing the test person. Leakage needs capacity to show up.

Uncertainty: the 95% intervals come from a bootstrap that resamples **people**, not frames, because
frames from one person are not independent.

## The transformer, piece by piece

1. **Patches.** The 64 x 32 frame is cut into 8 x 8 tiles: 8 x 4 = 32 patches. Each tile (64 numbers) is
   flattened and passed through one linear layer to make a 64-dimensional token. `PatchEmbed.patchify`
   does this with reshape and permute, and a test checks the tiles are exact.
2. **[CLS] token.** One extra learned token is prepended. After the blocks, its vector alone is fed to the
   classifier, so it has to gather information from every patch through attention.
3. **Positional embedding.** Attention by itself ignores order (a test proves it: shuffle the tokens and the
   outputs shuffle the same way). A learned position vector per token tells the model *where* on the mat a
   patch is, which matters: pressure at the shoulders means something different from pressure at the feet.
4. **Multi-head self-attention** (`pvit/attention.py`):
   - One linear layer makes queries, keys and values for every token.
   - Split into 4 heads of 16 dimensions each, so different heads can attend to different things.
   - Scores = Q K^T / sqrt(16). Dividing by sqrt(d_k) keeps the scores at unit scale. Without it,
     large dot products push softmax into a near one-hot regime where gradients vanish.
   - Softmax over keys, so each query's weights sum to 1 (tested).
   - Weighted sum of values, merge the heads, final linear layer.
   - Tested to match PyTorch's own `scaled_dot_product_attention` and `nn.MultiheadAttention` to 1e-6.
5. **Pre-LN blocks.** LayerNorm comes *before* attention and the MLP, with residual connections around both.
   This keeps a clean identity path through the network and trains stably at small scale without warm-up tricks.
6. **Size.** 4 blocks, 64 dimensions, about 0.14M parameters, small enough for an edge device.

## Training

- AdamW (weight decay 0.05), one-cycle learning rate, label smoothing 0.1, gradient clipping at 1.0.
- Hyperparameters are fixed in advance and never tuned on the held-out person, which would leak the test set.
- **One change after the first run:** at 14 epochs the ViT got only 87% of its own training frames right, so it was
  underfitting. It was rerun for 40 epochs (99% training accuracy on the leaky split). The decision used training
  accuracy only, and both runs are reported. Result on unseen people: 85.2% to 90.7%.
- **Calibration.** At 14 epochs the model was *under*confident (mean confidence 61%, accuracy 85%, ECE 0.24).
  Label smoothing, dropout, heavy augmentation and a short schedule all push confidence down. At 40 epochs ECE is 0.069.
- **Augmentations grounded in the physics of the mat:**
  - Shift by up to 2 sensors, because people do not lie in exactly the same place.
  - Gain jitter of +/-15%, because sensors drift and calibration differs.
  - **Mirror with label swap.** Mirroring a right-side frame gives a left-side frame, so the label is
    swapped too (position 2 becomes 3, 4 becomes 6, 13 becomes 14, and so on). This doubles the side-lying
    data honestly.

## Ablations (what each piece is worth)

- **No positional embedding.** Tests whether location on the mat matters.
- **No mirror augmentation.** Tests the physics-based augmentation.
- **CNN with a similar parameter budget.** The fair baseline. CNNs have locality built in, while a ViT has to
  learn it, so on small data a CNN is a serious competitor. Report the comparison honestly, whichever way it goes.
- **Logistic regression on pooled pixels.** A floor: how much does depth add at all?

Results, 17 positions, unseen people, all at 14 epochs:

| Variant | Accuracy |
|---|---|
| CNN | 89.7% |
| ViT | 85.2% |
| ViT, no mirror augmentation | 82.1% (mirror is worth 3.1 points) |
| Logistic regression | 79.9% |
| ViT, no positional embedding | 51.4% (location is worth 34 points) |

The CNN wins at equal budget. With 40 epochs the ViT reaches 90.7%, but the CNN was not rerun for 40 epochs
and the intervals overlap, so the honest claim is "comparable", not "better".

Errors sit between neighbouring angles: bed inclines 15/16/17 (30°, 45°, 60°) and the left wedges 6/7 (30° vs 60°).
Position 16 has the lowest recall (58%), mostly predicted as 15.

## Attention rollout

`attention_rollout` multiplies the head-averaged attention matrices (plus the residual identity) through
the layers to estimate how much each patch feeds into [CLS]. `figures/attention_rollout.png` shows it on a
held-out person, next to the mean pressure for each posture.

**Result:** rollout is close to uniform (largest patch 3.3% against 2.8% if spread evenly). The residual identity
added at every layer dominates in a 4-block model, so rollout does not single out body regions here. Say so in an
interview rather than reading meaning into it. Last-layer attention or gradient-based maps would be the next thing to try.

## Questions an interviewer may ask

- *Why a ViT for such a small image?* To learn attention from first principles on a sensor WATER actually
  uses. Then measure it against a CNN instead of assuming it wins.
- *What does attention give you here?* Every patch can use every other patch in one layer, so the model can
  relate shoulder load to hip load directly. A CNN needs depth to connect distant regions.
- *Why did the CNN beat your ViT?* At 14 epochs the ViT was underfitting. A CNN has locality built in, so it
  learns faster on 13 people. Trained longer, the ViT reaches 90.7%, level with the CNN within the intervals.
- *Why didn't the leaky split inflate results at first?* The 14-epoch model could not memorise its training
  frames. Once it could (40 epochs), the random split read 98.3% against 90.7% on new people.
- *Why is position 16 so hard?* It is a 45° bed incline between 30° and 60°. On the mat, neighbouring
  inclines shift weight only slightly. Temporal context or a regression target for angle would help.
- *What would you do with more data or time?* Use temporal attention across consecutive frames to track
  turning, train self-supervised on unlabelled nights, and calibrate per person with a few labelled frames.
