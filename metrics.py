import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2
import torch
import lpips

loss_fn = lpips.LPIPS(net='vgg')

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    return 20 * np.log10(255.0 / np.sqrt(mse))

def calculate_ssim(img1, img2):
    ssim_score = ssim(cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY),
                      cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY),
                      data_range=255)
    return ssim_score

def calculate_lpips(img1, img2):
    img1_tensor = lpips.im2tensor(img1)
    img2_tensor = lpips.im2tensor(img2)
    distance = loss_fn(img1_tensor, img2_tensor)
    return distance.item()
