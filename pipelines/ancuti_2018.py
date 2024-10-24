import os

import cv2
from methods.gamma_comps import gamma_correction
from methods.sharpening import unsharp_mask
from methods.wb_comps import comp_for_channel, gray_world
from methods.weight_maps import saturation_weight, laplacian_contrast_weight, saliency_weight
from .ancuti_2018_gui import Ancuti2018Gui


class Ancuti2018:
    def __init__(self, gui):
        # GUI STUFF
        self.gui = gui
        self.toolbox = Ancuti2018Gui(gui)

        # PROPERTIES
        self.image_original = None
        self.image_wb_precomp = None
        self.image_wb = None
        self.image_gamma = None
        self.image_sharp = None
        self.image_lapw_g = None
        self.image_salw_g = None
        self.image_satw_g = None
        self.image_lapw_s = None
        self.image_salw_s = None
        self.image_satw_s = None
        self.image_final_result = None

        self.value_wb_alpha_red = self.toolbox.slider_wb_alpha_red.value() / 10.0
        self.value_wb_alpha_blue = self.toolbox.slider_wb_alpha_blue.value() / 10.0
        self.value_gamma = self.toolbox.slider_gamma.value() / 10.0
        self.value_sharp_sigma = self.toolbox.slider_sharp_sigma.value()
        self.value_sharp_strength = self.toolbox.slider_sharp_strength.value() / 100.0

        self.toolbox.slider_wb_alpha_red.valueChanged.connect(self.set_wb_alpha_red)
        self.toolbox.slider_wb_alpha_blue.valueChanged.connect(self.set_wb_alpha_blue)
        self.toolbox.slider_gamma.valueChanged.connect(self.set_gamma)
        self.toolbox.slider_sharp_sigma.valueChanged.connect(self.set_sharp_sigma)
        self.toolbox.slider_sharp_strength.valueChanged.connect(self.set_sharp_strength)
        self.toolbox.combo_see_stage.currentTextChanged.connect(self.switch_stage_displayed)

    def set_wb_alpha_red(self, value):
        self.value_wb_alpha_red = value / 10.0
        self.toolbox.label_wb_alpha_red.setText(str(value / 10.0))

    def set_wb_alpha_blue(self, value):
        self.value_wb_alpha_blue = value / 10.0
        self.toolbox.label_wb_alpha_blue.setText(str(value / 10.0))

    def set_gamma(self, value):
        self.value_gamma = value / 10.0
        self.toolbox.label_gamma.setText(str(value / 10.0))

    def set_sharp_sigma(self, value):
        self.value_sharp_sigma = value
        self.toolbox.label_sharp_sigma.setText(str(value))

    def set_sharp_strength(self, value):
        self.value_sharp_strength = value / 100.0
        self.toolbox.label_sharp_strength.setText(str(value / 100.0))

    def process_image(self, img):
        if img is None:
            return

        self.image_original = img
        img_norm = cv2.normalize(img, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        result = img_norm.copy()

        if self.toolbox.check_wb_precomp.isChecked():
            result = comp_for_channel('red', img_norm, alpha=self.value_wb_alpha_red)
            result = comp_for_channel('blue', result, alpha=self.value_wb_alpha_blue)
            self.image_wb_precomp = result.copy()

        if self.toolbox.check_wb.isChecked():
            result = gray_world(result)
            self.image_wb = result.copy()

        if self.toolbox.check_gamma_sharp_msf.isChecked():
            self.image_gamma = gamma_correction(result, self.value_gamma)
            self.image_sharp = unsharp_mask(result, self.value_sharp_sigma, self.value_sharp_strength)

            self.image_lapw_g = laplacian_contrast_weight(self.image_gamma)
            self.image_satw_g = saturation_weight(self.image_gamma)
            self.image_salw_g = saliency_weight(self.image_gamma)

            self.image_lapw_s = laplacian_contrast_weight(self.image_sharp)
            self.image_satw_s = saturation_weight(self.image_sharp)
            self.image_salw_s = saliency_weight(self.image_sharp)

        self.image_final_result = result
        self.switch_stage_displayed("Final Result")
        self.toolbox.combo_see_stage.setCurrentText("Final Result")
        return self.image_final_result

    def save_results(self, save_path, name: str, extension: str):
        os.makedirs(f"{save_path}/{name}", exist_ok=True)
        path_prefix = f"{save_path}/{name}/{name}"

        image_attrs = {
            "image_original": "original",
            "image_wb_precomp": "wb_precomp",
            "image_wb": "wb",
            "image_gamma": "gamma",
            "image_sharp": "sharp",
            "image_lapw_g": "lapw_g",
            "image_satw_g": "satw_g",
            "image_salw_g": "salw_g",
            "image_lapw_s": "lapw_s",
            "image_satw_s": "satw_s",
            "image_salw_s": "salw_s",
            "image_final_result": "final_result"
        }

        idx = 1
        for attr, suffix in image_attrs.items():
            image = getattr(self, attr, None)
            if image is not None:
                full_path = f"{path_prefix}_{idx:02d}_{suffix}.{extension}"
                cv2.imwrite(full_path, cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
                idx += 1

    def switch_stage_displayed(self, value):
        to_display = None

        if value == "Original":
            to_display = self.image_original
        elif value == "After White Balance Pre-comp":
            to_display = self.image_wb_precomp
        elif value == "After White Balance":
            to_display = self.image_wb
        elif value == "After Gamma":
            to_display = self.image_gamma
        elif value == "After Sharpening":
            to_display = self.image_sharp
        elif value == "Laplacian Contrast Weight Map (Gamma)":
            to_display = self.image_lapw_g
        elif value == "Saliency Weight Map (Gamma)":
            to_display = self.image_salw_g
        elif value == "Saturation Weight Map (Gamma)":
            to_display = self.image_satw_g
        elif value == "Laplacian Contrast Weight Map (Sharpening)":
            to_display = self.image_lapw_s
        elif value == "Saliency Weight Map (Sharpening)":
            to_display = self.image_salw_s
        elif value == "Saturation Weight Map (Sharpening)":
            to_display = self.image_satw_s
        elif value == "Final Result":
            to_display = self.image_final_result

        self.gui.make_display_img(to_display, 'right')
