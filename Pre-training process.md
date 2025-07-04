# How to Train from the Official MAE Model to Our Customized Pre-trained Version

This guide explains how to start from the [official MAE model](https://github.com/facebookresearch/mae) and train a customized decoder while keeping the encoder frozen.

---

## 1. ✅ Select the MAE Model Version

Start with the official GitHub repo from Meta AI:
👉 **[MAE Official Repo](https://github.com/facebookresearch/mae)**

You can choose one of the following model variants depending on your resources:

* `mae_vit_base_patch16` — 12-layer encoder, 768-dim embedding
* `mae_vit_large_patch16` — 24-layer encoder, 1024-dim embedding
* `mae_vit_huge_patch14` — 32-layer encoder, 1280-dim embedding

---

## 2. ⚙️ Customize the MAE Architecture

To modify the decoder (e.g., use 512 embedding dim and 8 layers), edit the `model_mae.py` file. You can define a new model like this:

```python
def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
```

This allows you to control both the encoder and decoder configurations.

---

## 3. 🧊 Freeze the Encoder

In the same file (`model_mae.py`), freeze the encoder layers to prevent updates during training. After `self.initialize_weights()`, add:

```python
for param in self.patch_embed.parameters():
    param.requires_grad = False
for param in self.norm.parameters():
    param.requires_grad = False
for param in self.blocks.parameters():
    param.requires_grad = False
```

This ensures only the decoder is being updated.

---

## 4. 🧹 Remove Decoder Masking and Use MSE Loss

The default MAE decoder reconstructs only masked patches. To adapt it:

* **Remove** the masking logic in the decoder (i.e., reconstruct all patches).
* Use the decoder output to compute **MSE loss** against the full input image.

This allows the decoder to learn clean reconstruction and eliminate the original MAE pretext bias.

---

## 5. 📦 Our Pre-trained Model (Frozen Encoder + Trained Decoder)

After applying the steps above (custom decoder, frozen encoder, full-image loss), you can use our ready-to-use pre-trained weights:

📥 **Pre-trained MAE Checkpoint (resume/fine-tune):**
[Download from Google Drive](https://drive.google.com/file/d/16YnXfUeqBbSprhWV1OygriAsr2y9cCcf/view?usp=sharing)
