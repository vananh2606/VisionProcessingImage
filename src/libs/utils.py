from PyQt6.QtWidgets import QDialogButtonBox
from PyQt6.QtGui import QIcon, QPixmap, QImage, QAction
from PyQt6.QtCore import Qt

import cv2


class struct(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def newIcon(icon):
    return QIcon(":/" + icon)


def ndarray2pixmap(arr):
    if len(arr.shape) == 2:
        rgb = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    h, w, channel = rgb.shape
    qimage = QImage(rgb.data, w, h, channel * w, QImage.Format.Format_RGB888)
    pixmap = QPixmap.fromImage(qimage)
    return pixmap


def newDialogButton(parent, texts, slots, icons, orient=Qt.Orientation.Vertical):
    bb = QDialogButtonBox(orient, parent)
    for txt, slot, icon in zip(texts, slots, icons):
        but = bb.addButton(txt, QDialogButtonBox.ButtonRole.ApplyRole)
        but.setToolTip(txt)
        but.setObjectName("NewDialogButton")

        if slot is not None:
            but.clicked.connect(slot)

        if icon is not None:
            but.setIcon(newIcon(icon))
    return bb


def addActions(menu, actions):
    for act in actions:
        if isinstance(act, QAction):
            menu.addAction(act)
        else:
            menu.addMenu(act)


def newAction(
    parent, text, slot=None, shortcut=None, icon=None, tooltip=None, enabled=True
):
    a = QAction(text, parent)
    if icon is not None:
        a.setIcon(newIcon(icon))

    if shortcut is not None:
        a.setShortcut(shortcut)

    if slot is not None:
        a.triggered.connect(slot)

    if tooltip is not None:
        a.setToolTip(tooltip)

    a.setEnabled(enabled)
    return a
