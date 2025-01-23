import sys

sys.path.append("libs/")
sys.path.append("gui/")

from PyQt6.QtWidgets import QApplication
from mainwindow import MainWindow


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":

    main()
