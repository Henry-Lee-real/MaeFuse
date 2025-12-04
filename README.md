# MaeFuse
Official implementation of **MaeFuse: Transferring Omni Features with Pretrained Masked Autoencoders for Infrared and Visible Image Fusion via Guided Training** (TIP 2025)

[![arXiv 2404.11016](https://img.shields.io/badge/arXiv-2404.11016-b31b1b?logo=arXiv&logoColor=white&style=flat)](https://arxiv.org/abs/2404.11016)
[![IEEE TIP | Accept](https://img.shields.io/badge/IEEE%20TIP-Accept-00629B?logo=ieee&logoColor=white&style=flat)](https://ieeexplore.ieee.org/document/10893688)


Any questions can be consulted -> (Email:lijiayang.cs@gmail.com)

> 🚀 **NEWS:** Our TPAMI 2025 paper *"Towards Unified Semantic and Controllable Image Fusion: A Diffusion Transformer Approach"* is now released — code & models are available at [DiTFuse](https://github.com/Henry-Lee-real/DiTFuse).


## **Core Concept**: 

The mainstream approach often uses downstream tasks to drive fusion, resulting in more explicit object information in the fusion results. However, this leads to increased complexity and overfitting issues. Our core idea is that such complex downstream tasks are unnecessary. Instead, using a pre-trained encoder with high-level semantic information, such as MAE, can solve all problems effectively. The Fig.1 below explains this well. While we use MAE as the pre-trained encoder, you could also use other encoders with better performance, such as VAE.

![image](https://github.com/user-attachments/assets/8ee7b995-bd99-4dd5-b795-267c4b945aca)


Fig.1 Visualization results of the average fusion of feature vectors from different layers of the two modalities.

## Version I

![image](https://github.com/Henry-Lee-real/MaeFuse/assets/92620880/945d5ac0-5f88-4363-a34b-c8321276ba06)

## Version II
version 2 will be launched soon. Looking forward to your ⭐！

![演示文稿1](https://github.com/Henry-Lee-real/MaeFuse/assets/92620880/6144c130-d623-491c-b376-09ec9adb5cbd)

## How to use

**Setup**: Ensure you have Python 3.10 installed. Use the following command to initialize the environment:

```bash
pip install -r requirements.txt
```
**Pre-train Weight for MAE**(resume): https://drive.google.com/file/d/16YnXfUeqBbSprhWV1OygriAsr2y9cCcf/view?usp=sharing

📖 [Pre-training Process Guidance](./Pre-training-process.md)

**Test Weight**: https://drive.google.com/file/d/18N6tn78VztQOvobVWu6J-RJHo3jsBKkk/view?usp=sharing


**Note**: The dataset directory specified in `--address` must contain two subdirectories named `vi` and `ir` that contain visible and infrared images respectively.

To use this script, you need to provide the following command-line arguments:

1. `--checkpoint`: Path to the model checkpoint file (e.g., `final_new_60.pth`).
2. `--address`: Path to the dataset directory (e.g., `TNO_dataset`).
3. `--output`: Path to the folder where output images will be saved.

### Train Command
**Note**: For details on training code usage, refer to the internal documentation in `train.py`.
```bash
python train.py
```
📖 [Training Process Guidance](./training.md)
### Test Command

```bash
python test_fusion.py --checkpoint path_to_weight --address path_to_dataset --output path_to_output
```

### Description of Arguments

- `--checkpoint`: This argument specifies the file path to the model checkpoint, which is used to load the pre-trained model for image fusion.
- `--address`: This argument specifies the directory containing the dataset, which should include both visible and infrared images.
- `--output`: This argument specifies the directory where the fused output images will be saved. If the directory does not exist, it will be created automatically.

### Acknowledgements

We sincerely thank **Yuan Tu (NUE)** for raising many helpful issues and suggestions about the code implementation, which enabled us to refine the project and make it easier for the community to reproduce our results.


### Citation
```
@ARTICLE{10893688,
  author={Li, Jiayang and Jiang, Junjun and Liang, Pengwei and Ma, Jiayi and Nie, Liqiang},
  journal={IEEE Transactions on Image Processing}, 
  title={MaeFuse: Transferring Omni Features With Pretrained Masked Autoencoders for Infrared and Visible Image Fusion via Guided Training}, 
  year={2025},
  volume={34},
  number={},
  pages={1340-1353},
  keywords={Feature extraction;Visualization;Training;Image fusion;Data mining;Transformers;Semantics;Deep learning;Lighting;Image color analysis;Image fusion;vision transformer;masked autoencoder;guided training},
  doi={10.1109/TIP.2025.3541562}
}
```


