# MaeFuse
Official implementation of **MaeFuse: Transferring Omni Features with Pretrained Masked Autoencoders for Infrared and Visible Image Fusion via Guided Training**

*arxiv link:* https://arxiv.org/abs/2404.11016

## Abstract
> In this research, we introduce MaeFuse, a novel autoencoder model designed for infrared and visible image fusion (IVIF). The existing approaches for image fusion often rely on training combined with downstream tasks to obtain high-level visual information, which is effective in emphasizing target objects and delivering impressive results in visual quality and task-specific applications. MaeFuse, however, deviates from the norm. Instead of being driven by downstream tasks, our model utilizes a pretrained encoder from Masked Autoencoders (MAE), which facilities the omni features extraction for low-level reconstruction and high-level vision tasks, to obtain perception friendly features with a low cost. In order to eliminate the domain gap of different modal features and the block effect caused by the MAE encoder, we further develop a guided training strategy. This strategy is meticulously crafted to ensure that the fusion layer seamlessly adjusts to the feature space of the encoder, gradually enhancing the fusion effect. It facilitates the comprehensive integration of feature vectors from both infrared and visible modalities, preserving the rich details inherent in each. MaeFuse not only introduces a novel perspective in the realm of fusion techniques but also stands out with impressive performance across various public datasets.

## Compare

![image](https://github.com/Henry-Lee-real/MaeFuse/assets/92620880/945d5ac0-5f88-4363-a34b-c8321276ba06)
