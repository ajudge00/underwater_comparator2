import cv2
import numpy as np


def laplacian_contrast_weight(img: np.ndarray):
    assert img.dtype == np.float32

    # Step 1: Compute Laplacian with higher intensity for edges and texture (range -1 to 1)
    laplacian_edges = np.abs(cv2.normalize(
        cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_32F, ksize=7),
        None, -1.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F))

    # Step 2: Compute Laplacian with softer contrast (range 0 to 1)
    laplacian_base = cv2.normalize(
        cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_32F, ksize=3),
        None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)

    # Step 3: Combine the two by blending them, adjusting weights as needed
    contrast_weight = 0.4 * laplacian_base + 0.6 * laplacian_edges

    # Invert the weight to match the expected format
    return 1 - contrast_weight


def saturation_weight(img: np.ndarray):
    """
    W_Sat = √( (1/3) * [(R_k − L_k )^2 +(G_k − L_k )^2 +(B_k − L_k )^2])
    :param img: BGR input image (float32)
    :return: Saturation weight map
    """
    assert img.dtype == np.float32

    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    L = img_lab[:, :, 0] / 100.0

    B, G, R = cv2.split(img)

    R = R.astype(np.float32)
    G = G.astype(np.float32)
    B = B.astype(np.float32)

    W_Sat = cv2.sqrt(
        (1 / 3) * ((R - L) ** 2 + (G - L) ** 2 + (B - L) ** 2)
    )

    W_Sat = cv2.normalize(W_Sat, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)

    return 1 - cv2.cvtColor(W_Sat, cv2.COLOR_GRAY2BGR)


def saliency_weight(img: np.ndarray):
    assert img.dtype == np.float32

    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

    mean_image_feature_vector = img_lab.mean(axis=(0, 1))

    binomial_kernel_1d = np.array([1, 4, 6, 4, 1], dtype=np.float32)
    binomial_kernel_1d /= binomial_kernel_1d.sum()
    img_blurred = cv2.sepFilter2D(img_lab, -1, binomial_kernel_1d, binomial_kernel_1d)

    W_Sal = np.zeros(img_lab.shape[:2], dtype=np.float32)

    # for i in range(img_lab.shape[0]):
    #     for j in range(img_lab.shape[1]):
    #         W_Sal[i, j] = np.linalg.norm(mean_image_feature_vector - img_blurred[i, j], cv2.NORM_L2)

    diff = img_blurred - mean_image_feature_vector
    W_Sal = np.sqrt(np.sum(diff ** 2, axis=2))

    return cv2.normalize(W_Sal, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)


def get_saliency_ft(img: np.ndarray):
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    mean_val = np.mean(img, axis=(0, 1))

    im_blurred = cv2.GaussianBlur(img, (5, 5), 0)

    sal = np.linalg.norm(mean_val - im_blurred, axis=2)
    sal_max = np.max(sal)
    sal_min = np.min(sal)
    sal = ((sal - sal_min) / (sal_max - sal_min))

    sal = cv2.normalize(sal, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)
    return sal
