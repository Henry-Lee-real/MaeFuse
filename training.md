# Training Guide

This document explains the key training arguments in `train.py`, especially how to use different checkpoints and the meaning of each `--training_mode` option for multi-stage training.

---

## 1. Checkpoint Arguments

### `--resume`

```python
parser.add_argument(
    '--resume',
    default='output_decoder_4_640/checkpoint-50.pth',
    help='Pre-training ckpt for MAE (4-layer decoder)'
)
```
Purpose:
Path to a MAE pre-training checkpoint, typically containing the encoder + a 4-layer decoder.

Typical use cases:

Initialize training from a pretrained MAE model.

Resume training of the decoder-based stage from a previously saved checkpoint.

Note:
This argument is mainly used to restore MAE + decoder. It may not include weights for later fusion modules such as CFM or MFM.

--fusion_weight
python
复制代码
parser.add_argument(
    '--fusion_weight',
    default='path/to/fusion/layer/wegt',
    help='Checkpoint for fusion layer (CFM only)'
)
Purpose:
Path to a checkpoint that contains only the fusion layer / CFM + MFM weights.

Intuition:
Think of this as a lightweight checkpoint, separate from the full model.

--all_weight
python
复制代码
parser.add_argument(
    '--all_weight',
    default='path/to/fusion/all/wegt',
    help='Checkpoint for the entire model (decoder + CFM + MFM, etc.)'
)
Purpose:
Path to a checkpoint that saves all model parameters, including decoder, CFM, MFM, and possibly other modules.

Source:
During training, the code will save full-model checkpoints. These can later be used for multi-stage training.

Typical use cases:

After finishing Stage 1 (e.g., use the resume ckpt), you use this full checkpoint as the initialization for Stage 2.

For any later experiment where you want to fully restore the model state from a previous stage.

Rule of thumb

--resume: MAE / decoder-oriented pretraining checkpoint.

--fusion_weight: fusion-layer-only checkpoint.

--all_weight: full-model checkpoint, recommended for multi-stage loading.

2. --training_mode Overview
python
复制代码
parser.add_argument(
    '--training_mode',
    default="train_decoder_CFM_MFM",
    type=str,
    choices=[
        'train_CFM_mean',
        'train_CFM_fusion',
        'train_MFM_mean_CFM_lock',
        'train_MFM_fusion_CFM_lock',
        'train_MFM_mean_CFM_open',
        'train_MFM_fusion_CFM_open',
        'train_decoder_only',
        'train_decoder_MFM',
        'train_decoder_CFM_MFM',
    ]
)
We mainly consider three modules:

Decoder: MAE decoder (reconstructs images from latent features).

CFM: Cross Fusion Module (feature-level fusion between two images/modalities).

MFM: Mask / Multi-stage Fusion Module (additional fusion / refinement module).

Each training_mode controls:

Which modules are trainable (Decoder / CFM / MFM).

Which loss is used:

mean = feature mean-based loss.

fusion = fusion-based loss (usually at pixel / reconstruction level).

Whether CFM is updated:

*_CFM_lock = CFM is frozen.

*_CFM_open = CFM is trainable.

Below is the detailed explanation for each mode.

2.1 CFM-only training modes
train_CFM_mean
Train CFM only, with a mean-based loss.

Trainable modules:

CFM ✅

Decoder ❌ (frozen)

MFM ❌

Loss:
A mean-based loss using the features from the two input images. CFM learns to fuse features by matching the mean statistics.

Use case:
Initial, relatively simple training stage to make CFM learn a stable, average-style fusion behavior.

train_CFM_fusion
Train CFM only, with a fusion loss through the decoder and pixel-level supervision.

Trainable modules:

CFM ✅

Decoder ❌ (frozen)

MFM ❌

Loss:

Fused features are passed through the (frozen) decoder to reconstruct images.

A fusion / reconstruction loss is computed at the pixel level.

Gradients flow through the decoder into CFM, but only CFM is updated, decoder remains fixed.

Use case:
Refine CFM using a stronger, task-oriented pixel-level loss while keeping the decoder stable.

2.2 Decoder / Decoder+MFM modes (CFM is frozen)
train_decoder_only
Train decoder only, without CFM or MFM.

Trainable modules:

Decoder ✅

CFM ❌

MFM ❌

Use case:

Pretraining the decoder itself so that it has good reconstruction capability.

Later stages can introduce CFM and MFM on top of a strong decoder.

train_decoder_MFM
Train decoder + MFM, CFM is frozen (or not used), with a fusion loss.

Trainable modules:

Decoder ✅

MFM ✅

CFM ❌

Loss:
Fusion-based loss (typically at pixel level) that supervises the interaction between MFM and decoder.

Use case:

Given fixed CFM outputs or a setup without CFM, jointly train MFM and the decoder to improve fusion quality.

2.3 Full joint training: Decoder + CFM + MFM
train_decoder_CFM_MFM
Train Decoder, CFM, and MFM together, with a fusion loss.

Trainable modules:

Decoder ✅

CFM ✅

MFM ✅

Loss:
Fusion-based loss (e.g., pixel reconstruction or task-specific fusion loss).

Use case:

Final joint fine-tuning stage.

After the modules have been reasonably initialized (e.g., via previous stages), perform end-to-end training to maximize performance.

2.4 MFM training with CFM locked or open
These modes focus on training MFM, while controlling whether CFM is updated and whether the loss is mean-based or fusion-based.

train_MFM_mean_CFM_lock
Train MFM only, CFM is frozen, using a mean-based loss.

Trainable modules:

MFM ✅

CFM ❌ (locked)

Decoder: depends on your implementation (often fixed or partially updated).

Loss:
Mean-based feature loss (similar to train_CFM_mean, but focusing on MFM).

Use case:

Keep a previously trained CFM fixed.

Let MFM learn a simple, mean-based objective on top of stable CFM outputs.

train_MFM_fusion_CFM_lock
Train MFM only, CFM is frozen, using a fusion loss.

Trainable modules:

MFM ✅

CFM ❌ (locked)

Loss:
Fusion-based loss (typically via decoder and pixel-level supervision). Only MFM is updated.

Use case:

Refine MFM under a stronger, task-relevant fusion loss while preserving the CFM weights.

train_MFM_mean_CFM_open
Train MFM + CFM, using a mean-based loss.

Trainable modules:

MFM ✅

CFM ✅

Loss:
Mean-based feature loss.

Use case:

Jointly adjust MFM and CFM under a relatively simple objective.

Useful after initial pretraining if you want both modules to adapt together but still use a stable, mean-based supervision.

train_MFM_fusion_CFM_open
Train MFM + CFM, using a fusion loss.

Trainable modules:

MFM ✅

CFM ✅

Loss:
Fusion-based loss (e.g., decoder-based pixel reconstruction).

Use case:

Strongest joint optimization stage for MFM + CFM.

Recommended after having good initial checkpoints (e.g., from --all_weight).
