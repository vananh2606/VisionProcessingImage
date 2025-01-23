from PyQt6.QtGui import *
from PyQt6.QtCore import *

import numpy as np


# color shape
DEFAULT_FILL_COLOR = QColor(128, 128, 255, 100)
DEFAULT_SELECT_FILL_COLOR = QColor(255, 100, 100, 50)
DEFAULT_VISIBLE_FILL_COLOR = QColor(128, 128, 0, 0)
DEFAULT_VERTEX_FILL_COLOR = QColor(255, 255, 255, 0)
DEFAULT_VERTEX_SELECT_FILL_COLOR = QColor(255, 255, 0, 255)
DEFAULT_SELECT_COLOR = QColor(255, 0, 0, 255)


class Shape(object):
    RADIUS = 7
    THICKNESS = 7
    FONT_SIZE = 30
    MIN_WIDTH = 20

    def __init__(self, label=None):
        super(Shape, self).__init__()
        self.points = []
        self.selected = False
        self.visible = False
        self.corner = None
        self.label = label
        self.image_debug = None
        # shape parameter
        self.config = None
        self.scale = 1.0
        self._b_lock = False
        self._b_hide = False
        #
        self.vertex_fill_color = DEFAULT_VERTEX_FILL_COLOR
        self.vertex_select_fill_color = DEFAULT_VERTEX_SELECT_FILL_COLOR
        self.fill_color = DEFAULT_FILL_COLOR
        self.select_color = DEFAULT_SELECT_COLOR
        self.select_fill_color = DEFAULT_SELECT_FILL_COLOR
        self.visible_fill_color = DEFAULT_VISIBLE_FILL_COLOR

    @property
    def lock(self):
        return self._b_lock

    @lock.setter
    def lock(self, val: bool):
        self._b_lock = val

    @property
    def hide(self):
        return self._b_hide

    @hide.setter
    def hide(self, val: bool):
        self._b_hide = val

    def drawVertex(self, path, i):
        d = Shape.RADIUS
        point = self.points[i]
        if self.corner is not None and i == self.corner:
            path.addRect(point.x() - d, point.y() - d, 2 * d, 2 * d)
        elif self.visible:
            path.addEllipse(point, d / 2, d / 2)
        else:
            path.addEllipse(point, d / 2.0, d / 2.0)

    def paint(self, painter, s=1):
        if self.hide:
            return
        self.scale = s
        color = QColor("green")
        lw = Shape.THICKNESS
        painter.setPen(QPen(color, lw))
        line_path = QPainterPath()
        vertex_path = QPainterPath()

        line_path.moveTo(self.points[0])

        for i, p in enumerate(self.points):
            line_path.lineTo(p)
            self.drawVertex(vertex_path, i)

        line_path.lineTo(self.points[0])
        #  draw rect
        painter.drawPath(line_path)
        # draw label
        if self.label is not None:
            fs = Shape.FONT_SIZE
            font = QFont("Arial", fs)
            painter.setFont(font)
            painter.drawText(int(self[0].x()) - 1, int(self[0].y()) - 1, self.label)

        #  fill
        color = (
            self.vertex_select_fill_color
            if (self.visible or self.corner is not None)
            else self.vertex_fill_color
        )
        color = self.select_color if self.selected else color
        painter.fillPath(vertex_path, color)

        # color = self.visible_fill_color if (self.visible or self.corner is not None) else self.fill_color
        # color = self.select_fill_color if self.selected else color
        # painter.fillPath(line_path, color)

    def translate_(self, v: QPointF):
        self.points = [p + v for p in self.points]

    def move(self, v: QPointF):
        if not self.lock:
            self.points = [p + v for p in self.points]
        pass

    def copy(self):
        shape = Shape(label=self.label + "_copy")
        shape.points = self.points
        shape.visible = self.visible
        shape.corner = self.corner
        shape.config = self.config
        shape.translate_(QPointF(50.0, 50.0))
        return shape

    def contain(self, pos):
        x, y = pos.x(), pos.y()
        tl = self.points[0]
        br = self.points[2]
        x1, y1 = tl.x(), tl.y()
        x2, y2 = br.x(), br.y()
        return x1 < x < x2 and y1 < y < y2

    def get_corner(self, pos, epsilon=10):
        for i in range(len(self.points)):
            d = self.distance(pos, self.points[i])
            if d < epsilon:
                self.corner = i
                return True
        self.corner = None
        return False

    def distance(self, p1, p2):
        p = p2 - p1
        return np.sqrt(p.x() ** 2 + p.y() ** 2)

    def dis_to(self, pos):
        x, y = pos.x(), pos.y()
        tl = self.points[0]
        br = self.points[2]
        dx = min([np.abs(x - tl.x()), np.abs(x - br.x())])
        dy = min([np.abs(y - tl.y()), np.abs(y - br.y())])
        if self.contain(pos):
            return min(dx, dy)
        else:
            return -min(dx, dy)

    def change(self, v):
        if self.lock:
            return
        points = self.points
        corner = self.corner
        R = QRectF(self.points[0], self.points[2])
        pos = self.points[corner] + v

        if corner == 0:
            R.setTopLeft(pos)
        elif corner == 1:
            R.setTopRight(pos)
        elif corner == 2:
            R.setBottomRight(pos)
        elif corner == 3:
            R.setBottomLeft(pos)

        ret, points = self.get_points(R)
        if ret:
            self.points = points

    def get_points(self, r=QRectF()):
        pos1 = r.topLeft()
        pos2 = r.topRight()
        pos3 = r.bottomRight()
        pos4 = r.bottomLeft()

        width = pos3.x() - pos1.x()
        height = pos3.y() - pos1.y()

        if width > Shape.MIN_WIDTH and height > Shape.MIN_WIDTH:
            ret = True
        else:
            ret = False
        return ret, [pos1, pos2, pos3, pos4]

    @property
    def cvBox(self):
        tl = self.points[0]
        br = self.points[2]
        x, y = tl.x(), tl.y()
        w, h = br.x() - x, br.y() - y

        x, y, w, h = list(map(int, [x, y, w, h]))
        x = max(x, 0)
        y = max(y, 0)
        return [x, y, w, h]

    def __len__(self):
        return len(self.points)

    def __getitem__(self, key):
        return self.points[key]

    def __setitem__(self, key, value):
        self.points[key] = value


if __name__ == "__main__":
    pass
