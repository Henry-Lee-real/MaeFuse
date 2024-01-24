import cv2
import numpy as np
import os
from tqdm import *
from skimage.metrics import structural_similarity as ssim
from scipy.stats import entropy

def calculate_scd(image1, image2):
    """计算两个图像之间的结构内容差异"""
    mean1 = np.mean(image1)
    mean2 = np.mean(image2)
    return mean1 / mean2 if mean2 != 0 else 0


def calculate_entropy(image):
    """计算图像的熵"""
    hist = cv2.calcHist([image], [0], None, [256], [0,256])
    hist_norm = hist.ravel()/hist.sum()
    return entropy(hist_norm, base=2)

def calculate_std(image):
    """计算图像的标准差"""
    return np.std(image)

def mutual_information(hgram):
    """计算互信息"""
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

def calculate_mi(image1, image2):
    """计算两个图像之间的互信息"""
    hgram, _, _ = np.histogram2d(image1.ravel(), image2.ravel(), bins=20)
    return mutual_information(hgram)

def find_image(path, filename):
    """在给定路径中寻找具有相同名称但可能不同格式的图像"""
    basename, _ = os.path.splitext(filename)
    for file in os.listdir(path):
        if file.startswith(basename) and (file.endswith(".jpg") or file.endswith(".png")):
            return cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
    return None

def main():
    path_a_vi = 'RoadScene/vi'#'MSRS-main/train/vi'
    path_a_ir = 'RoadScene/ir'
    path_b = 'C:/Users/Henry/Desktop/trans_swin/SeA_road'

    metrics = {'EN': [], 'SD': [], 'MI_vi': [], 'MI_ir': [], 'SSIM_vi': [], 'SSIM_ir': [], 'SCD_vi': [], 'SCD_ir': []}
    total_files = len([name for name in os.listdir(path_b) if name.endswith(".jpg") or name.endswith(".png")])

    for filename in tqdm(os.listdir(path_b)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_b = cv2.imread(os.path.join(path_b, filename), cv2.IMREAD_GRAYSCALE)
            #img_b = cv2.resize(img_b, (640, 480))
            img_vi = find_image(path_a_vi, filename)
            img_ir = find_image(path_a_ir, filename)

            if img_vi is not None and img_ir is not None:
                metrics['EN'].append(calculate_entropy(img_b))
                metrics['SD'].append(calculate_std(img_b))
                metrics['MI_vi'].append(calculate_mi(img_b, img_vi))
                metrics['MI_ir'].append(calculate_mi(img_b, img_ir))
                metrics['SSIM_vi'].append(ssim(img_b, img_vi))
                metrics['SSIM_ir'].append(ssim(img_b, img_ir))
                metrics['SCD_vi'].append(calculate_scd(img_b, img_vi))
                metrics['SCD_ir'].append(calculate_scd(img_b, img_ir))

    # 计算平均值
    for key in metrics:
        metrics[key] = np.mean(metrics[key])

    return metrics

if __name__ == "__main__":
    results = main()
    print(results)

