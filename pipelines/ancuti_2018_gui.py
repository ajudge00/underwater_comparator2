from PyQt5.QtWidgets import QSlider, QLabel, QCheckBox, QComboBox


class Ancuti2018Gui:
    def __init__(self, gui):
        self.slider_wb_alpha_red = gui.findChild(QSlider, "slider_wb_alpha_red")
        self.slider_wb_alpha_blue = gui.findChild(QSlider, "slider_wb_alpha_blue")
        self.slider_gamma = gui.findChild(QSlider, "slider_gamma")
        self.slider_sharp_sigma = gui.findChild(QSlider, "slider_sharp_sigma")
        self.slider_sharp_strength = gui.findChild(QSlider, "slider_sharp_strength")

        self.label_wb_alpha_red = gui.findChild(QLabel, "label_wb_alpha_red")
        self.label_wb_alpha_blue = gui.findChild(QLabel, "label_wb_alpha_blue")
        self.label_gamma = gui.findChild(QLabel, "label_gamma")
        self.label_sharp_sigma = gui.findChild(QLabel, "label_sharp_sigma")
        self.label_sharp_strength = gui.findChild(QLabel, "label_sharp_strength")

        self.check_wb_precomp = gui.findChild(QCheckBox, "check_wb_pc")
        self.check_wb = gui.findChild(QCheckBox, "check_wb")
        self.check_gamma_sharp_msf = gui.findChild(QCheckBox, "check_gamma_sharp_msf")

        self.combo_see_stage = gui.findChild(QComboBox, "combo_see_stage")

        self.view_choices = {
            "original": ["Original"],
            "wbpc": ["After White Balance Pre-comp"],
            "wb": ["After White Balance"],
            "gsmsf": ["After Gamma",
                      "After Sharpening",
                      "Laplacian Contrast Weight Map (Gamma)",
                      "Saliency Weight Map (Gamma)",
                      "Saturation Weight Map (Gamma)",
                      "Laplacian Contrast Weight Map (Sharpening)",
                      "Saliency Weight Map (Sharpening)",
                      "Saturation Weight Map (Sharpening)"],
            "fr": ["Final Result"]
        }

        self.view_choices_flattened = [
            "Original", "After White Balance Pre-comp", "After White Balance", "After Gamma", "After Sharpening",
            "Laplacian Contrast Weight Map (Gamma)", "Saliency Weight Map (Gamma)",
            "Saturation Weight Map (Gamma)", "Laplacian Contrast Weight Map (Sharpening)",
            "Saliency Weight Map (Sharpening)", "Saturation Weight Map (Sharpening)", "Final Result"
        ]

        self.check_wb_precomp.stateChanged.connect(
            lambda: self.change_combo_see_stage("wbpc", self.check_wb_precomp.isChecked())
        )
        self.check_wb.stateChanged.connect(
            lambda: self.change_combo_see_stage("wb", self.check_wb.isChecked())
        )
        self.check_gamma_sharp_msf.stateChanged.connect(
            lambda: self.change_combo_see_stage("gsmsf", self.check_gamma_sharp_msf.isChecked())
        )

        self.original_choice_groups = list(self.view_choices.keys())
        self.current_choice_groups = self.original_choice_groups.copy()
        self.build_combo_see_stage()

    def change_combo_see_stage(self, choice: str, checked: bool):
        if checked:
            i = self.original_choice_groups.index(choice)
            self.current_choice_groups.insert(i, choice)
        else:
            self.current_choice_groups.remove(choice)

        self.build_combo_see_stage()

    def build_combo_see_stage(self):
        self.combo_see_stage.clear()
        for group in self.current_choice_groups:
            self.combo_see_stage.addItems(self.view_choices[group])

        self.combo_see_stage.setCurrentIndex(self.combo_see_stage.count() - 1)
