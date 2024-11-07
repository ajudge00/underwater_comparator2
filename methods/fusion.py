import enum
import cv2


class FusionType(enum.Enum):
    NAIVE = 0,
    MULTI_SCALE = 1


def apply_fusion(fusion_type: FusionType,
                 input1, input2, weight1, weight2,
                 levels: int = 3):
    if fusion_type == FusionType.NAIVE:
        return naive_fusion(input1, input2, weight1, weight2)
    elif fusion_type == FusionType.MULTI_SCALE:
        return multi_scale_fusion(input1, input2, weight1, weight2, levels)


def naive_fusion(input1, input2, weight1, weight2):
    return weight1 * input1 + weight2 * input2


def multi_scale_fusion(input1, input2, weight1, weight2, levels=3):
    Weight1_pyr = gaussian_pyramid(weight1, levels)
    Weight2_pyr = gaussian_pyramid(weight2, levels)

    R1, G1, B1 = cv2.split(input1)
    R2, G2, B2 = cv2.split(input2)

    R1_pyr, G1_pyr, B1_pyr = [laplacian_pyramid(chan, levels) for chan in (R1, G1, B1)]
    R2_pyr, G2_pyr, B2_pyr = [laplacian_pyramid(chan, levels) for chan in (R2, G2, B2)]

    R_fused = [Weight1_pyr[k] * R1_pyr[k] + Weight2_pyr[k] * R2_pyr[k] for k in range(levels)]
    G_fused = [Weight1_pyr[k] * G1_pyr[k] + Weight2_pyr[k] * G2_pyr[k] for k in range(levels)]
    B_fused = [Weight1_pyr[k] * B1_pyr[k] + Weight2_pyr[k] * B2_pyr[k] for k in range(levels)]

    R = reconstruct_pyramid(R_fused)
    G = reconstruct_pyramid(G_fused)
    B = reconstruct_pyramid(B_fused)

    fusion = cv2.merge((R, G, B))
    return fusion


def gaussian_pyramid(img, levels):
    pyr = [img]

    for i in range(1, levels):
        img = cv2.pyrDown(img)
        pyr.append(img)

    return pyr


def laplacian_pyramid(img, levels):
    g_pyr = gaussian_pyramid(img, levels)
    l_pyr = []

    for i in range(levels - 1):
        size = (g_pyr[i].shape[1], g_pyr[i].shape[0])
        # cv2.imshow(f"{i}", g_pyr[i] - cv2.pyrUp(g_pyr[i + 1]))
        l_pyr.append(g_pyr[i] - cv2.pyrUp(g_pyr[i + 1], dstsize=size))

    l_pyr.append(g_pyr[-1])

    return l_pyr


def reconstruct_pyramid(pyr):
    img = pyr[-1]
    for level in reversed(pyr[:-1]):
        size = (level.shape[1], level.shape[0])
        img = cv2.pyrUp(img, dstsize=size) + level
    return img
