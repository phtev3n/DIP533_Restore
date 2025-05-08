import numpy as np
import cv2
from PIL import Image

def compress_jpeg(image_array, quality):
    img = Image.fromarray(image_array)
    img.save('temp.jpg', quality=quality, subsampling=0)
    compressed_img = Image.open('temp.jpg')
    return np.array(compressed_img)

def gaussian_filter(image_array, sigma=1.5):
    return cv2.GaussianBlur(image_array, (5, 5), sigma)

def median_filter(image_array, kernel_size=3):
    return cv2.medianBlur(image_array, kernel_size)

def bilateral_filter(image_array, diameter=9, sigma_color=20, sigma_space=5):
    return cv2.bilateralFilter(image_array, diameter, sigma_color, sigma_space)


