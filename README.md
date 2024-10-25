# MaeFuse
Official implementation of **MaeFuse: Transferring Omni Features with Pretrained Masked Autoencoders for Infrared and Visible Image Fusion via Guided Training**

*arxiv link:* https://arxiv.org/abs/2404.11016

if you want to test on your datasets(email:lijiayang.cs@gmail.com) you can get the wight for the code!

## Abstract
> In this research, we introduce MaeFuse, a novel autoencoder model designed for infrared and visible image fusion (IVIF). The existing approaches for image fusion often rely on training combined with downstream tasks to obtain high-level visual information, which is effective in emphasizing target objects and delivering impressive results in visual quality and task-specific applications. MaeFuse, however, deviates from the norm. Instead of being driven by downstream tasks, our model utilizes a pretrained encoder from Masked Autoencoders (MAE), which facilities the omni features extraction for low-level reconstruction and high-level vision tasks, to obtain perception friendly features with a low cost. In order to eliminate the domain gap of different modal features and the block effect caused by the MAE encoder, we further develop a guided training strategy. This strategy is meticulously crafted to ensure that the fusion layer seamlessly adjusts to the feature space of the encoder, gradually enhancing the fusion effect. It facilitates the comprehensive integration of feature vectors from both infrared and visible modalities, preserving the rich details inherent in each. MaeFuse not only introduces a novel perspective in the realm of fusion techniques but also stands out with impressive performance across various public datasets.

## Compare

![image](https://github.com/Henry-Lee-real/MaeFuse/assets/92620880/945d5ac0-5f88-4363-a34b-c8321276ba06)

## Version II
version 2 will be launched soon. Looking forward to your ⭐！

![演示文稿1](https://github.com/Henry-Lee-real/MaeFuse/assets/92620880/6144c130-d623-491c-b376-09ec9adb5cbd)

## How to use

**Note**: The dataset directory specified in `--address` must contain two subdirectories named `vi` and `ir` that contain visible and infrared images respectively.

To use this script, you need to provide the following command-line arguments:

1. `--checkpoint`: Path to the model checkpoint file (e.g., `final_new_60.pth`).
2. `--address`: Path to the dataset directory (e.g., `TNO_dataset`).
3. `--output`: Path to the folder where output images will be saved.

### Example Command

```bash
python script.py --checkpoint final_new_60.pth --address TNO_dataset --output TNO_55_final_2
```

### Description of Arguments

- `--checkpoint`: This argument specifies the file path to the model checkpoint, which is used to load the pre-trained model for image fusion.
- `--address`: This argument specifies the directory containing the dataset, which should include both visible and infrared images.
- `--output`: This argument specifies the directory where the fused output images will be saved. If the directory does not exist, it will be created automatically.

The script will read images from the provided dataset directory, perform the fusion operation using the loaded model, and save the results to the specified output folder.





