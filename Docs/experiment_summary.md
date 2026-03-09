# Experiment & Training Summary

Summary of 52 TensorBoard training experiments conducted locally but excluded from the repository due to their binary nature and size (336 MB).

---

## Overview

| Detail | Value |
|---|---|
| Total experiments | 52 |
| Date range | Jan 7, 2025 (6:12 PM) -- Jan 9, 2025 (10:46 AM) |
| Duration | ~40 hours of experimentation |
| Hardware | GQM-IT-DSK-017 (CUDA GPU) |
| Metrics logged | 8 scalar losses |
| Longest run | 94,100 steps (~19 hours wall-clock) |

## Logged Metrics

| Metric | Description |
|---|---|
| `Loss/generator` | Adversarial hinge loss for the generator |
| `Loss/discriminator` | Adversarial hinge loss for the discriminator |
| `Loss/cycle` | Cycle consistency loss (L1) |
| `Loss/identity` | Identity mapping loss (L1) |
| `Loss/f0` | Fundamental frequency (pitch) RMSE loss |
| `Loss/stft` | Multi-resolution STFT reconstruction loss |
| `Loss/cosine` | Stable cosine distance loss |
| `Loss/divergence` | KL divergence on speaker embeddings |

---

## Experiment Breakdown

| Category | Count | Description |
|---|---|---|
| 0 steps (startup only) | 34 | Configuration tests / failed to start training |
| 1--100 steps | 2 | Very short test runs |
| 101--1,000 steps | 7 | Short training runs |
| 1,001--10,000 steps | 8 | Medium training runs |
| 10,001+ steps | 1 | Long training run (the main experiment) |

Most experiments (34/52) were iterative debugging runs that never completed a training step. The remaining 18 produced actual training metrics.

---

## Two Distinct Training Phases

### Phase 1: Full Loss Function (Jan 7 -- Jan 8 morning)
- **Experiments**: `20250107_185201` through `20250108_120918`
- **Metrics**: All 8 losses active (including cosine and STFT)
- **Generator loss**: Extremely high (7,600 -- 768,000) due to the cosine embedding loss dominating
- **Observation**: The cosine distance loss (values ~768,000) overwhelmed other losses, making training unstable

### Phase 2: Refactored Loss Function (Jan 8 afternoon)
- **Experiments**: `20250108_144850` through `20250108_152755`
- **Metrics**: 4 losses only (generator, discriminator, cycle, identity)
- **Generator loss**: Near zero or negative (good -- means generator fools discriminator)
- **Observation**: Removing cosine and STFT losses from the generator objective stabilized training significantly

---

## Best Performing Experiments

| Metric | Best Value | Experiment | Steps |
|---|---|---|---|
| Loss/generator | -0.7455 | experiment_20250108_152755 | 94,100 |
| Loss/discriminator | 0.5761 | experiment_20250107_192004 | 4,200 |
| Loss/cycle | 0.0017 | experiment_20250108_152755 | 94,100 |
| Loss/identity | 0.0026 | experiment_20250108_152755 | 94,100 |
| Loss/f0 | 0.6103 | experiment_20250107_181756 | 0 |
| Loss/stft | 41.4340 | experiment_20250107_185849 | 1,900 |
| Loss/divergence | 2.5999 | experiment_20250108_103601 | 7,800 |

---

## Star Experiment: `experiment_20250108_152755`

The longest and best-performing run (94,100 steps, ~19 hours):

### Learning Curve

| Metric | Start | 25% | 50% | 75% | Final |
|---|---|---|---|---|---|
| Loss/generator | 1.3881 | -0.5854 | -0.7298 | -0.7436 | **-0.7455** |
| Loss/discriminator | 0.9993 | 0.7500 | 0.7500 | 0.7500 | **0.7500** |
| Loss/cycle | 0.5788 | 0.0507 | 0.0064 | 0.0022 | **0.0017** |
| Loss/identity | 0.7877 | 0.1139 | 0.0138 | 0.0041 | **0.0026** |

### Key Observations
- Generator loss went negative, indicating the generator consistently fools the discriminator
- Discriminator loss stabilized at ~0.75 (GAN equilibrium -- discriminator is ~50/50 on real vs fake)
- Cycle consistency loss dropped by **334x** (0.579 to 0.0017)
- Identity loss dropped by **305x** (0.788 to 0.0026)
- Clear convergence with no signs of mode collapse

---

## Failed Experiments (NaN)

5 experiments produced NaN values (numerical instability), all from Jan 8 around 2:30--2:43 PM:

- `experiment_20250108_143456` (100 steps)
- `experiment_20250108_143918` (100 steps)
- `experiment_20250108_144026` (100 steps)
- `experiment_20250108_144150` (200 steps)
- `experiment_20250108_144320` (100 steps)

These likely represent hyperparameter tuning attempts that diverged quickly.

---

## Trend Analysis Across Experiments

| Metric | Early Avg | Late Avg | Improvement |
|---|---|---|---|
| Loss/generator | 276,562 | 0.22 | ~100% (architecture change) |
| Loss/f0 | 8.74 | 1.96 | 77.6% |
| Loss/cycle | 0.55 | 0.17 | 70.1% |
| Loss/identity | 0.72 | 0.33 | 54.6% |
| Loss/discriminator | 0.66 | 0.77 | -15.7% (expected -- GAN equilibrium) |

---

## Checkpoints

The `checkpoints/vc1/` directory is **empty**. No model weights were saved from these experiments. The training code saves checkpoints to `/home/goquest/VC48/checkpoints/vc1/` (a different machine), so any saved weights reside there rather than in this local copy.

---

## Training Configuration (from `train.py`)

| Parameter | Value |
|---|---|
| Batch size | 32 |
| Learning rate | 0.0001 |
| Optimizer | Adam (betas: 0.5, 0.999) |
| Epochs | 1,000 |
| Sample rate | 48 kHz |
| Mel bins | 128 |
| FFT size | 1,920 |
| Hop length | 480 |
| Chunk size | 24,000 samples (0.5s) |
| Source speaker | AVANI |
| Target speaker | ADESH |
