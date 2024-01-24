import cv2
import numpy as np
import math
from tqdm import tqdm
import os
from scipy.signal import convolve2d

# EN 计算函数
def imageEn(image):
    tmp = [0] * 256
    k = 0
    res = 0
    img = np.array(image)
    for i in range(len(img)):
        for j in range(len(img[i])):
            val = img[i][j]
            tmp[val] += 1
            k += 1
    tmp = [x / k for x in tmp]
    for i in range(len(tmp)):
        if tmp[i] != 0:
            res -= tmp[i] * (math.log(tmp[i]) / math.log(2.0))
    return res

# SSIM 计算函数
def matlab_style_gauss2D(shape=(3,3), sigma=0.5):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)

def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):
    if not im1.shape == im2.shape:
        raise ValueError("Input Images must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")
    M, N = im1.shape
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    window = matlab_style_gauss2D(shape=(win_size, win_size), sigma=1.5)
    window = window / np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1*im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2*im2, window, 'valid') - mu2_sq
    sigma12 = filter2(im1*im2, window, 'valid') - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return np.mean(np.mean(ssim_map))

# AG 计算函数
def average_gradient(image):
    # 计算x和y方向上的梯度
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # 计算梯度幅度
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # 计算平均梯度
    average_gradient = np.mean(gradient_magnitude)

    return average_gradient

# SF 计算函数
def spatialF(image):
    M, N = image.shape
    cf, rf = 0, 0
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            dx = float(image[i, j - 1]) - float(image[i, j])
            rf += dx ** 2
            dy = float(image[i - 1, j]) - float(image[i, j])
            cf += dy ** 2
    RF = math.sqrt(rf / (M * N))
    CF = math.sqrt(cf / (M * N))
    SF = math.sqrt(RF ** 2 + CF ** 2)
    return SF

# MI 计算函数
def getMI(im1, im2):
    hang, lie = im1.shape
    N = 256
    h = np.zeros((N, N))
    for i in range(hang):
        for j in range(lie):
            h[im1[i, j], im2[i, j]] += 1
    h /= np.sum(h)
    im1_marg = np.sum(h, axis=0)
    im2_marg = np.sum(h, axis=1)
    H_x = -np.sum([px * math.log2(px) for px in im1_marg if px != 0])
    H_y = -np.sum([py * math.log2(py) for py in im2_marg if py != 0])
    H_xy = -np.sum([p * math.log2(p) for p in h.flatten() if p != 0])
    MI = -H_xy + H_x + H_y
    return MI

def calculate_spatial_frequency(image):
    

    # 计算傅里叶变换
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)

    # 计算空间频率
    magnitude_spectrum = 20 * np.log(np.abs(f_shift))
    return magnitude_spectrum

def scd_similarity(image1, image2):
    # 计算两幅图像的空间频率
    sf1 = calculate_spatial_frequency(image1)
    sf2 = calculate_spatial_frequency(image2)

    # 计算相似度
    similarity = np.sum((sf1 - sf2) ** 2)
    return similarity

# 主程序
# 此处省略遍历文件夹和计算平均值的代码，因为我将接着编写 VIFF 和 FMI 的 Python 实现
def main(folder_A, folder_B):
    # 存储每种计算的结果
    results = {"EN": [], "SSIM": [], "AG": [], "SF": [], "MI": [], "SCD": []}#, "VIFF": [], "FMI": []}
    
    for filename in tqdm(os.listdir(folder_B)):
        # 加载文件夹B中的图片
        img_B = cv2.imread(os.path.join(folder_B, filename), 0)

        # 加载对应的文件夹A/vi和A/ir中的图片
        img_A_vi = cv2.imread(os.path.join(folder_A, "vi", filename), 0)
        img_A_ir = cv2.imread(os.path.join(folder_A, "ir", filename), 0)

        # 计算并存储每种度量
        results["EN"].append(imageEn(img_B))
        results["SSIM"].append(compute_ssim(img_B, img_A_vi) + compute_ssim(img_B, img_A_ir))
        results["AG"].append(average_gradient(img_B))
        results["SF"].append(spatialF(img_B))
        results["MI"].append(getMI(img_B, img_A_vi) + getMI(img_B, img_A_ir))
        results["SCD"].append(scd_similarity(img_B, img_A_vi) + scd_similarity(img_B, img_A_ir))
        # VIFF 和 FMI 的计算需要编写对应的Python函数
        # results["VIFF"].append(calculate_viff(...))
        # results["FMI"].append(calculate_fmi(...))

    # 计算平均值
    averages = {key: np.mean(val) for key, val in results.items()}
    return averages

if __name__ == "__main__":
    folder_A = "MSRS-main/test"
    folder_B = "MSRS_test_result_640_14_new_99+14"
    averages = main(folder_A, folder_B)
    print(averages)