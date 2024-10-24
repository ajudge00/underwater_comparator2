import cv2
import numpy as np


def saturation_weight(img: np.ndarray):
    """
    W_Sat = √( (1/3) * [(R_k − L_k )^2 +(G_k − L_k )^2 +(B_k − L_k )^2])
    :param img: BGR input image (float32)
    :return: Saturation weight map
    """
    assert img.dtype == np.float32

    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    L = img_lab[:, :, 0] / 255.0

    B, G, R = cv2.split(img)

    R = R.astype(np.float32)
    G = G.astype(np.float32)
    B = B.astype(np.float32)

    W_Sat = cv2.sqrt(
        (1 / 3) * ((R - L) ** 2 + (G - L) ** 2 + (B - L) ** 2)
    )

    return cv2.cvtColor(1 - W_Sat, cv2.COLOR_GRAY2BGR)


def laplacian_contrast_weight(img: np.ndarray):
    assert img.dtype == np.float32

    img_greyscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    W_Lap = cv2.Laplacian(img_greyscale, cv2.CV_32F)

    return cv2.cvtColor(1 - np.abs(W_Lap), cv2.COLOR_GRAY2BGR)


def saliency_weight(img: np.ndarray):
    assert img.dtype == np.float32

    binomial_kernel_1d = np.array([1, 4, 6, 4, 1], dtype=np.float32)
    binomial_kernel_1d /= binomial_kernel_1d.sum()

    img_blurred = cv2.sepFilter2D(img, -1, binomial_kernel_1d, binomial_kernel_1d)
    # img_blurred = cv2.GaussianBlur(img, (5, 5), 0)
    lab = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2Lab)

    L = lab[:, :, 0].astype(np.float32)
    A = lab[:, :, 1].astype(np.float32)
    B = lab[:, :, 2].astype(np.float32)

    lm = np.mean(L)
    am = np.mean(A)
    bm = np.mean(B)

    W_Sal = (L - lm) ** 2 + (A - am) ** 2 + (B - bm) ** 2

    W_Sal /= np.max(W_Sal)

    return W_Sal

    # assert img.dtype == np.float32
    #
    # binomial_kernel_1d = np.array([1, 4, 6, 4, 1], dtype=np.float32)
    # binomial_kernel_1d /= binomial_kernel_1d.sum()
    #
    # img_blurred = cv2.sepFilter2D((img * 255).astype(np.uint8), -1, binomial_kernel_1d, binomial_kernel_1d)
    # img_cielab = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2LAB)
    #
    # mean_l = cv2.mean(img_cielab[:, :, 0])[0]
    # mean_a = cv2.mean(img_cielab[:, :, 1])[0]
    # mean_b = cv2.mean(img_cielab[:, :, 2])[0]
    #
    # W_Sal = (cv2.subtract(img_cielab[:, :, 0], mean_l) ** 2 +
    #          cv2.subtract(img_cielab[:, :, 1], mean_a) ** 2 +
    #          cv2.subtract(img_cielab[:, :, 2], mean_b) ** 2)
    #
    # return W_Sal
