import enum
import numpy as np


class CompChannel(enum.Enum):
    COMP_RED = 0
    COMP_BLUE = 1


def comp_for_channel(channel: CompChannel, img: np.ndarray, alpha=1):
    """
    I_rc(x) = I_r(x) + α * (¯I_g − ¯I_r) * (1 − I_r(x)) * I_g(x)\n
    I_bc(x) = I_b(x) + α * (¯I_g − ¯I_b) * (1 − I_b(x)) * I_g(x)

    :param channel: The channel to compensate for (COMP_RED or COMP_BLUE)
    :param img: The image to be processed (has to be [0.0-1.0] float32)
    :param alpha: The strength of the compensation
    """

    avg_b = np.mean(img[:, :, 0])
    avg_g = np.mean(img[:, :, 1])
    avg_r = np.mean(img[:, :, 2])

    res = img.copy()

    if channel == CompChannel.COMP_RED:
        res[:, :, 2] = img[:, :, 2] + alpha * (avg_g - avg_r) * (1 - img[:, :, 2]) * img[:, :, 1]
    elif channel == CompChannel.COMP_BLUE:
        res[:, :, 0] = img[:, :, 0] + alpha * (avg_g - avg_b) * (1 - img[:, :, 0]) * img[:, :, 1]

    return res


def gray_world(img: np.ndarray):
    avg_b = np.mean(img[:, :, 0])
    avg_g = np.mean(img[:, :, 1])
    avg_r = np.mean(img[:, :, 2])

    # Tanulság: ez hülyeség volt, mert ilyen képeknél *mindig* a zöld
    # lesz a domináns csatorna, azaz alpha és beta *mindig* nagyobb lesz 1.0-nál
    # és a clippeléssel így mindig 1.0-t kapunk, ami 1-gyel szorzás lesz :/
    # alpha = min(1.0, avg_g / avg_r)
    # betha = min(1.0, avg_g / avg_b)

    alpha = avg_g / avg_r
    betha = avg_g / avg_b

    res = img.copy()

    res[:, :, 2] = res[:, :, 2] * alpha
    res[:, :, 0] = res[:, :, 0] * betha

    return res
