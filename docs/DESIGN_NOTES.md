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

## Attention rollout

`attention_rollout` multiplies the head-averaged attention matrices (plus the residual identity) through
the layers to estimate how much each patch feeds into [CLS]. `figures/attention_rollout.png` shows it on a
held-out person: which body regions the model relies on for each posture.

## Questions an interviewer may ask

- *Why a ViT for such a small image?* To learn attention from first principles on a sensor WATER actually
  uses. Then measure it against a CNN instead of assuming it wins.
- *What does attention give you here?* Every patch can use every other patch in one layer, so the model can
  relate shoulder load to hip load directly. A CNN needs depth to connect distant regions.
- *What would you do with more data or time?* Use temporal attention across consecutive frames to track
  turning, train self-supervised on unlabelled nights, and calibrate per person with a few labelled frames.
