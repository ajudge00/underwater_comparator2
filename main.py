import sys
from PyQt5.QtWidgets import QApplication
from gui import GUI


class App:
    def __init__(self, gui_path):
        self.gui = GUI(gui_path)
        self.gui.showMaximized()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    UIWindow = App("main.ui")
    app.exec_()
