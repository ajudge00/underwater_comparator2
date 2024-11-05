import os

import cv2

from methods import myutils
from methods.wb_comps import comp_for_channel, CompChannel, gray_world
from methods.gamma_comps import gamma_correction
from methods.weight_maps import *
from methods.sharpening import *

img_original = cv2.imread('../test_images/harang.jpg', cv2.IMREAD_COLOR)
img_original_f32 = myutils.convert_dtype(img_original, myutils.ConvertFlags.UINT8_TO_F32)


"""
White-Balance Pre-comp for red (and blue)
"""
img_wb_precomp = comp_for_channel(CompChannel.COMP_RED, img_original_f32, alpha=1.0)
img_wb_precomp = comp_for_channel(CompChannel.COMP_BLUE, img_wb_precomp, alpha=0.0)


"""
White-Balance Correction using the Gray-World Assumption
"""
img_wb = gray_world(img_wb_precomp)


"""
Input 1 (Gamma Correction) and its weight maps for the eventual Multi-Scale Fusion process
"""
input1 = gamma_correction(img_wb, 3.5)
input1_laplacian_wm = get_weight_map(input1, method=WeightMapMethods.LAPLACIAN)
input1_saliency_wm = get_weight_map(input1, method=WeightMapMethods.SALIENCY)
input1_saturation_wm = get_weight_map(input1, method=WeightMapMethods.SATURATION)


"""
Input 2 (Normalized Unsharp Masking) and its weight maps for the eventual Multi-Scale Fusion process
"""
input2 = norm_unsharp_mask(img_wb)
input2_laplacian_wm = get_weight_map(input2, method=WeightMapMethods.LAPLACIAN)
input2_saliency_wm = get_weight_map(input2, method=WeightMapMethods.SALIENCY)
input2_saturation_wm = get_weight_map(input2, method=WeightMapMethods.SATURATION)

"""
Normalized weight maps of Input1 and Input2
"""
input1_norm_wm, input2_norm_wm = normalize_weight_maps(
    input1_laplacian_wm, input1_saliency_wm, input1_saturation_wm,
    input2_laplacian_wm, input2_saliency_wm, input2_saturation_wm
)


"""
Images to display
"""
# cv2.imshow('Initial', img_original_f32)
# cv2.imshow('After Pre-comp', img_wb_precomp)
# cv2.imshow('After White Balance (Gray World)', img_wb)

# cv2.imshow('Input1 (Gamma)', input1)
# cv2.imshow('Laplacian Contrast Weight (Input1)', input1_laplacian_wm)
# cv2.imshow('Saliency Weight (Input1)', input1_saliency_wm)
# cv2.imshow('Saturation Weight (Input1)', input1_saturation_wm)
#
# cv2.imshow('Input2 (Normalized Unsharp Masking)', input2)
# cv2.imshow('Laplacian Contrast Weight (Input2)', input2_laplacian_wm)
# cv2.imshow('Saliency Weight (Input2)', input2_saliency_wm)
# cv2.imshow('Saturation Weight (Input2)', input2_saturation_wm)
#
cv2.imshow('Normalized Weight Map of Input1', input1_norm_wm)
cv2.imshow('Normalized Weight Map of Input2', input2_norm_wm)

filenames = {
    'input1': input1,
    'input1_laplacian': input1_laplacian_wm,
    'input1_saliency': input1_saliency_wm,
    'input1_saturation': input1_saturation_wm,
    'input1_normalized': input1_norm_wm,
    'input2': input2,
    'input2_laplacian': input2_laplacian_wm,
    'input2_saliency': input2_saliency_wm,
    'input2_saturation': input2_saturation_wm,
    'input2_normalized': input2_norm_wm,
}

# i = 0
# for filename, image in filenames.items():
#     # image = cv2.putText(image, filename, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#     image = (image * 255).astype(np.uint8)
#     path = 'results/' + "{:02d}".format(i) + f'_{filename}.jpg'
#
#     cv2.imwrite(path, image)
#     i += 1

cv2.waitKey(0)
cv2.destroyAllWindows()
