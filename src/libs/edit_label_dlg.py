from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QLineEdit,
    QListWidget,
    QDialogButtonBox,
    QVBoxLayout,
)
from PyQt6.QtGui import QCursor


class BoxEditLabel(QDialog):
    def __init__(self, title="QDialog", parent=None):
        super(BoxEditLabel, self).__init__(parent)
        self.setWindowTitle(title)

        BB = QDialogButtonBox
        bb = BB(BB.StandardButton.Ok | BB.StandardButton.Cancel)
        bb.rejected.connect(self.reject)
        bb.accepted.connect(self.accept)

        self.ln_name = QLineEdit()
        self.ln_name.setFocus()
        self.list_name = QListWidget()

        layout = QVBoxLayout(self)
        layout.addWidget(self.ln_name)
        layout.addWidget(bb)
        layout.addWidget(self.list_name)

        self.list_name.itemClicked.connect(self.itemClicked)
        self.list_name.itemDoubleClicked.connect(self.itemDoubleClicked)

    def itemClicked(self, item):
        self.ln_name.setText(item.text())

    def itemDoubleClicked(self, item):
        self.ln_name.setText(item.text())
        self.accept()

    def popUp(self, text="", names=[], bMove=False):
        self.list_name.clear()
        self.list_name.addItems(names)
        self.ln_name.setText(text)
        self.ln_name.setSelection(0, len(text))
        if bMove:
            self.move(QCursor.pos())
        return self.ln_name.text() if self.exec() else ""


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    win = BoxEditLabel()
    win.show()
    sys.exit(app.exec())
