import cv2
import numpy as np
from scipy.ndimage.filters import median_filter
from .myutils import convert_dtype, ConvertFlags


def num2(img: np.ndarray):
    gaussian_filtered = cv2.GaussianBlur(img, (5, 5), 0)
    diff = img - gaussian_filtered

    min_val = np.min(diff)
    max_val = np.max(diff)

    stretched_img = (diff - min_val) / (max_val - min_val)

    return (img + stretched_img) / 2


def norm_unsharp_mask_matlab(img: np.ndarray, sigma=20, N=30, gain=0.8):
    Igauss = img.copy()

    for _ in range(N):
        Igauss = cv2.GaussianBlur(Igauss, (0, 0), sigma)
        Igauss = np.minimum(img, Igauss)

    Norm = img - gain * Igauss

    Norm_eq = np.zeros_like(Norm)
    for n in range(3):
        Norm_eq[:, :, n] = cv2.equalizeHist((Norm[:, :, n] * 255).astype(np.uint8)) / 255.0

    Isharp = (img + Norm_eq) / 2
    return Isharp


def norm_unsharp_mask(img: np.ndarray):
    """
    Normalized unsharp mask\n
    S = (I + N {I − G ∗ I}) / 2\n
    where\n
    - G ∗ I is the Gaussian-filtered image and\n
    - N is the linear normalization (histogram stretching) operator.

    :param img: image to process
    :return: S
    """

    gaussian_filtered = cv2.GaussianBlur(img, (5, 5), 0)
    diff = img - gaussian_filtered
    normalized_diff = cv2.normalize(diff, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)

    return (img + normalized_diff) / 2


def unsharp(image, sigma, strength):
    image_mf = median_filter(image, sigma)
    lap = cv2.Laplacian(image, cv2.CV_32F)
    sharp = cv2.subtract(image, (cv2.multiply(strength, lap)))

    return sharp


def unsharp_mask(img: np.ndarray, sigma, strength):
    img_copy = img.copy()

    for i in range(3):
        img_copy[:, :, i] = unsharp(img[:, :, i], sigma, strength)

    return img_copy
