# import numpy 
# import utils
import os
import torchvision.transforms as transforms
# 图像文件所在的文件夹路径
import cv2
import torch
# image_folder = "MSRS-main/train/vi"  # 图像文件夹的路径



# # 获取文件夹中所有图片文件的名称（不包含文件类型）
# image_names = []

# for name in os.listdir(image_folder):
#     if name.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
#         image_name = os.path.splitext(name)[0]  # 获取不含扩展名的文件名
#         image_names.append(image_name)

# # 打印所有图片文件的名称
# for name in image_names:
#     print(name)
transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
img_vi = cv2.imread("test_fusion/vi.png")
out  = transform_train(torch.tensor(img_vi))
print(out)
print(type(out))
print(out.shape)




# original_imgs_path = 

# image_set_ir, batches = utils.load_dataset(original_imgs_path, batch_size)