# Training Guide

This document explains the major training arguments in **`train.py`**, including how to load different checkpoints and how to use each `--training_mode` option for multi-stage training.

---

# 1. Checkpoint Arguments

## 1.1 `--resume`

```python
parser.add_argument(
    '--resume',
    default='output_decoder_4_640/checkpoint-50.pth',
    help='Pre-training ckpt for MAE (4-layer decoder)'
)
```

**Purpose**
Checkpoint for **MAE pretraining**, typically containing:

* Encoder
* 4-layer decoder

**Use cases**

* Initialize training from a pretrained MAE model
* Resume decoder-based training from a saved checkpoint

**Notes**

* Contains **MAE + decoder** only
* Usually **does NOT** include CFM or MFM weights

---

## 1.2 `--fusion_weight`

```python
parser.add_argument(
    '--fusion_weight',
    default='path/to/fusion/layer/wegt',
    help='Checkpoint for fusion layer (CFM only)'
)
```

**Purpose**
Checkpoint containing only **fusion layers** (CFM or MFM).

**Notes**

* Lightweight checkpoint
* Used when training CFM/MFM independently

---

## 1.3 `--all_weight`

```python
parser.add_argument(
    '--all_weight',
    default='path/to/fusion/all/wegt',
    help='Checkpoint for the entire model (decoder + CFM + MFM, etc.)'
)
```

**Purpose**
A **full-model checkpoint**, containing:

* Decoder
* CFM
* MFM
* Other related modules

**Use cases**

* Load as initialization for **multi-stage training**
* Fully restore model state from a previous training stage

**Rule of thumb**

* `--resume` → MAE / decoder checkpoint
* `--fusion_weight` → fusion module only
* `--all_weight` → full-model checkpoint for multi-stage loading

---

# 2. `--training_mode` Overview

```python
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
```

We consider three modules:

* **Decoder** — MAE decoder
* **CFM** — Fusion module
* **MFM** — Multi-feature module

Each mode controls:

* Which modules are **trainable**
* Whether CFM is **locked** or **open**
* Whether the loss is **mean-based** or **fusion-based**

---

# 2.1 CFM-only Training Modes

## `train_CFM_mean`

**Trainable**

* CFM ✅
* Decoder ❌ (no use)
* MFM ❌ (no use)

**Loss**
Mean-based feature loss.

**Use case**
Simple initialization stage to stabilize CFM behavior.

## `train_CFM_fusion`

**Trainable**

* CFM ✅
* Decoder ❌ (no use)
* MFM ❌ (no use)

**Loss**
Fusion/pixel reconstruction loss via decoder
(grad flows through decoder but decoder is not updated).

**Use case**
Refine CFM with stronger supervision while keeping decoder stable.

---

# 2.2 Decoder / Decoder+MFM (CFM Frozen)

## `train_decoder_only`

**Trainable**

* Decoder ✅
* CFM ❌
* MFM ❌

**Use case**
Pretrain decoder for strong reconstruction capability.

## `train_decoder_MFM`

**Trainable**

* Decoder ✅
* MFM ✅
* CFM ❌

**Loss**
Fusion-based loss.

**Use case**
Train decoder and MFM jointly while keeping CFM frozen or unused.

---

# 2.3 Full Joint Training

## `train_decoder_CFM_MFM`

**Trainable**

* Decoder ✅
* CFM ✅
* MFM ✅

**Loss**
Fusion-based loss (pixel reconstruction, etc.)

**Use case**
Final end-to-end joint fine-tuning after all modules are initialized.

---

# 2.4 MFM Training (CFM Locked or Open)

## `train_MFM_mean_CFM_lock`

**Trainable**

* MFM ✅
* CFM ❌ (locked)

**Loss**
Mean-based loss.

**Use case**
Train MFM only on top of a fixed CFM.

## `train_MFM_fusion_CFM_lock`

**Trainable**

* MFM ✅
* CFM ❌ (locked)

**Loss**
Fusion/pixel reconstruction loss.

**Use case**
Refine MFM under stronger task-oriented supervision.

## `train_MFM_mean_CFM_open`

**Trainable**

* MFM ✅
* CFM ✅

**Loss**
Mean-based loss.

**Use case**
Joint MFM + CFM training under a simpler objective.

## `train_MFM_fusion_CFM_open`

**Trainable**

* MFM ✅
* CFM ✅

**Loss**
Fusion/pixel reconstruction loss.

**Use case**
Strongest optimization stage for MFM + CFM; recommended after loading a good `--all_weight` checkpoint.
