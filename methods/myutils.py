import enum
import numpy as np


class ConvertFlags(enum.Enum):
    UINT8_TO_F32 = 0
    F32_TO_UINT8 = 1


def convert_dtype(img: np.ndarray, convert_flag: ConvertFlags) -> np.ndarray:
    if convert_flag == ConvertFlags.UINT8_TO_F32:
        img_converted = img.copy()
        img_converted = (img_converted / 255.0).astype(np.float32)
        return img_converted
    elif convert_flag == ConvertFlags.F32_TO_UINT8:
        img_converted = img.copy()
        img_converted = (img_converted * 255.0).astype(np.uint8)
        return img_converted
