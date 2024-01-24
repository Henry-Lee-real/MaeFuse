import os
import shutil
import math

def sample_images(input_folder, output_folder, num_samples=300):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取输入文件夹中所有图像文件
    images = [file for file in os.listdir(input_folder) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    # 计算总数和步长
    total_images = len(images)
    step = max(1, math.ceil(total_images / num_samples))

    # 选择图片
    selected_images = images[::step][:num_samples]

    # 复制图片到输出文件夹
    for image in selected_images:
        shutil.copy(os.path.join(input_folder, image), os.path.join(output_folder, image))

# 使用示例
input_folder = 'LLVIP/ir'  # 替换为输入文件夹的路径
output_folder = 'LLVIP_300/ir'  # 替换为输出文件夹的路径
sample_images(input_folder, output_folder)
