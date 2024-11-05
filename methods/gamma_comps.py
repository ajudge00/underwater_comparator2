import cv2
import numpy as np
from methods import myutils


def create_gamma_lut(gamma):
    lut = np.arange(0, 256, 1, np.float32)
    lut = lut / 255.0
    lut = lut ** gamma
    lut = np.uint8(lut * 255.0)

    return lut


# def gamma_correction(img: np.ndarray, gamma: float):
#     # res = cv2.LUT((255 * img).astype(np.uint8), create_gamma_lut(gamma))
#     # return res.astype(np.float32) / 255
#
#     assert img.dtype == np.float32
#
#     res = myutils.convert_dtype(img.copy(), myutils.ConvertFlags.F32_TO_UINT8)
#     res = cv2.LUT(res, create_gamma_lut(gamma))
#     return myutils.convert_dtype(res, myutils.ConvertFlags.UINT8_TO_F32)


def gamma_correction(img: np.ndarray, gamma: float):
    assert img.dtype == np.float32

    res = img ** gamma
    return res
