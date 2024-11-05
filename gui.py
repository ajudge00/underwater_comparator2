import os.path

import cv2
import imutils
import numpy as np
from PyQt5 import uic, QtGui
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QMainWindow, QLabel, QPushButton, QComboBox, QStackedWidget, QFileDialog

from pipelines.ancuti_2018 import Ancuti2018
from pipelines.mohan_simon_2020 import MohanSimon2020


class GUI(QMainWindow):
    def __init__(self, gui_path):
        super(GUI, self).__init__()
        uic.loadUi(gui_path, self)

        self.combo_pipeline = self.findChild(QComboBox, "combo_pipeline")
        self.build_combo_pipeline()
        self.combo_pipeline.currentIndexChanged.connect(self.switch_pipeline)

        self.img_left = self.findChild(QLabel, "img_original")
        self.img_right = self.findChild(QLabel, "img_result")

        self.button_load = self.findChild(QPushButton, "button_load")
        self.button_load.clicked.connect(self.load_image)
        self.button_process = self.findChild(QPushButton, "button_process")
        self.button_process.clicked.connect(self.process_image)
        self.button_save = self.findChild(QPushButton, "button_save")
        self.button_save.clicked.connect(self.save_results)

        self.stacked_pipelines = self.findChild(QStackedWidget, "stacked_pipelines")
        self.pipeline_ancuti_2018 = Ancuti2018(self)
        self.pipeline_mohan_simon_2020 = MohanSimon2020(self)
        self.pipelines = [self.pipeline_ancuti_2018, self.pipeline_mohan_simon_2020]
        self.current_pipeline = self.pipeline_ancuti_2018

        self.image_name = ""
        self.image_extension = ""
        self.image_original = None

    def build_combo_pipeline(self):
        pipeline_choices = ["Ancuti et al. 2018", "Mohan-Simon 2020"]
        self.combo_pipeline.clear()
        self.combo_pipeline.addItems(pipeline_choices)
        self.combo_pipeline.setCurrentIndex(0)

    def switch_pipeline(self, value):
        self.stacked_pipelines.setCurrentIndex(value)
        self.current_pipeline = self.pipelines[value]

    def load_image(self):
        try:
            filename = QFileDialog.getOpenFileName(filter="Képfájlok (*.png *.jpg *.jpeg *.bmp)")[0]
            self.image_name, self.image_extension = os.path.basename(filename).split('.')
            self.image_original = cv2.imread(filename)
            self.make_display_img(self.image_original, 'left')
        except Exception as e:
            print("Nem lett kép kiválasztva, vagy egyéb hiba.")

    def process_image(self):
        result = self.current_pipeline.process_image(self.image_original)
        self.make_display_img(result, 'right')

    def save_results(self):
        save_path = QFileDialog.getExistingDirectory(self, "Save results to...")
        self.current_pipeline.save_results(save_path, self.image_name, self.image_extension)

    def make_display_img(self, img, side):
        if img is None:
            print("img is None!")
            return

        if img.dtype == np.float32:
            # img = (img * 255).astype(np.uint8)
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        image_resized = imutils.resize(img, width=720)
        frame = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        image_displayed = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)

        if side == 'left':
            self.img_left.setPixmap(QtGui.QPixmap.fromImage(image_displayed))
        elif side == 'right':
            self.img_right.setPixmap(QtGui.QPixmap.fromImage(image_displayed))
