import sys
from shape import *
from edit_label_dlg import BoxEditLabel
from utils import *
import resources

from functools import partial

from PyQt6.QtWidgets import (
    QLabel,
    QMainWindow,
    QMenu,
    QWidget,
    QApplication,
    QPushButton,
    QMessageBox,
)

import os

# cursor
CURSOR_DEFAULT = Qt.CursorShape.ArrowCursor
CURSOR_POINT = Qt.CursorShape.PointingHandCursor
CURSOR_DRAW = Qt.CursorShape.CrossCursor
CURSOR_DRAW_POLYGON = Qt.CursorShape.SizeAllCursor
CURSOR_MOVE = Qt.CursorShape.ClosedHandCursor
CURSOR_GRAB = Qt.CursorShape.OpenHandCursor


class Canvas(QLabel):
    mouseMoveSignal = pyqtSignal(QPointF)
    newShapeSignal = pyqtSignal(int)
    editShapeSignal = pyqtSignal(str)
    deleteShapeSignal = pyqtSignal(int)
    moveShapeSignal = pyqtSignal(int)
    drawShapeSignal = pyqtSignal(QRectF)
    changeShapeSignal = pyqtSignal(int)
    selectedShapeSignal = pyqtSignal(int)
    zoomSignal = pyqtSignal(float)
    actionSignal = pyqtSignal(str)
    applyConfigSignal = pyqtSignal()

    def __init__(self, parent=None, bcontext_menu=True, benable_drawing=True):
        super().__init__(parent)
        self.setObjectName("Canvas")
        self.bcontext_menu = bcontext_menu
        self.picture: QPixmap = None
        # self.picture = QPixmap(1280,1020)
        self.painter = QPainter()
        self.scale = 1
        self.org = QPointF()
        self.moving = False
        self.edit = False
        self.drawing = False
        self.highlight = False
        self.wheel = False
        self.current_pos = QPointF()
        self.current = None
        self.win_start_pos = QPointF()
        self.start_pos = QPointF()
        self.start_pos_moving = QPointF()

        self.line1 = [QPointF(), QPointF()]
        self.line2 = [QPointF(), QPointF()]

        self.text_pixel_color = "BGR:"
        self.shapes = []
        # self.dict_shapes = {}
        self.idVisible = None
        self.idSelected = None
        self.idCorner = None
        self.benable_drawing = benable_drawing
        self.label_path = "classes.txt"
        self.labels = self.load_label()
        self.last_label = ""
        self.boxEditLabel = BoxEditLabel("Enter shape name", self)
        # ========
        self.contextMenu = QMenu()
        action = partial(newAction, self)
        add_grid = action("Create Grid", None, "", "", "Create grid for")
        copy = action("copy", self.copyShape, "", "copy", "copy shape")

        lock = action("Lock", self.change_lock, "", "lock", "Lock/Unlock shape")
        lock_all = action(
            "Lock All", self.change_lock_all, "", "lock", "Lock/Unlock all shapes"
        )

        # hide = action("Hide", self.change_hide,"","hide","Hide/Show shape")
        hide_all = action(
            "Hide All", self.change_hide_all, "", "lock", "Hide/Show all shapes"
        )

        edit = action("edit", self.editShape, "", "edit", "edit shape")
        delete = action("delete", self.deleteShape, "", "delete", "delete shape")
        delete_all = action("delete all", self.delete_all, "", "", "delete shape")

        self.actions = struct(
            add_grid=add_grid,
            copy=copy,
            edit=edit,
            delete=delete,
            delete_all=delete_all,
            lock=lock,
            lock_all=lock_all,
            hide_all=hide_all,
        )
        addActions(self.contextMenu, [add_grid])
        self.contextMenu.addSeparator()
        addActions(self.contextMenu, [lock, lock_all])
        addActions(self.contextMenu, [hide_all])
        self.contextMenu.addSeparator()
        addActions(self.contextMenu, [edit, copy, delete, delete_all])

        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.popUpMenu)

        #
        self.tool_zoom = QWidget(self)
        self.label_pos = QLabel(self)
        self.label_pos.setStyleSheet(
            "QLabel{background-color:rgba(128, 128, 128, 150); color:white; font:bold 12px}"
        )
        self.label_pos.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self.zoom_buttons = newDialogButton(
            self.tool_zoom,
            ["", "", "", ""],
            [
                lambda: self.zoom_manual(1.2),
                lambda: self.zoom_manual(0.8),
                lambda: self.fit_window(),
                self.on_show_full_screen,
            ],
            icons=["zoom_in", "zoom_out", "zoom_fit", "full_screen"],
            orient=Qt.Orientation.Horizontal,
        ).buttons()

        btn: QPushButton = None
        for btn in self.zoom_buttons:
            btn.setObjectName("ZoomDialogButton")

        # =======
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.WheelFocus)
        # ==============

    def load_label(self):
        try:
            with open(self.label_path, "r") as file:
                lines = file.readlines()
                labels = [line.strip("\r\n") for line in lines]
                return labels
        except:
            return []

    def show_grid(self, b_show=True):
        shape: Shape = None
        for shape in self:
            if "P" in shape.label:
                if b_show:
                    shape.hide = False
                else:
                    shape.hide = True

    def change_hide(self):
        index = self.idSelected
        s: Shape = None

        if index is not None:
            s: Shape = self[index]
            if s.hide:
                s.hide = False
            else:
                s.hide = True

    def change_hide_all(self):
        s: Shape = None
        self.cancel_selected()
        if self.actions.hide_all.text() == "Hide All":
            self.actions.hide_all.setText("Show All")
            for s in self:
                s.hide = True
        else:
            self.actions.hide_all.setText("Hide All")
            for s in self:
                s.hide = False

    def change_lock(self):
        index = self.idSelected
        s: Shape = None

        if index is not None:
            s: Shape = self[index]
            if s.lock:
                s.lock = False
            else:
                s.lock = True

    def change_lock_all(self):
        s: Shape = None

        if self.actions.lock_all.text() == "Lock All":
            self.actions.lock_all.setText("UnLock All")
            for s in self:
                s.lock = True
        else:
            self.actions.lock_all.setText("Lock All")
            for s in self:
                s.lock = False

    def set_benable_drawing(self, enable):
        self.benable_drawing = enable
        if self.benable_drawing:
            self.actions.disable_drawing.setText("Disable drawing")
        else:
            self.actions.disable_drawing.setText("Enable drawing")

    def setEnabledActions(self, enable):
        self.actions.add_grid.setEnabled(enable)
        self.actions.copy.setEnabled(enable)
        self.actions.edit.setEnabled(enable)
        self.actions.delete.setEnabled(enable)
        self.actions.delete_all.setEnabled(enable)
        self.actions.lock.setEnabled(enable)
        pass

    def popUpMenu(self):
        if self.idSelected is None:
            self.setEnabledActions(False)
        else:
            self.setEnabledActions(True)
            s: Shape = self[self.idSelected]
            if s.lock:
                self.actions.lock.setText("UnLock")
            else:
                self.actions.lock.setText("Lock")

        if self.bcontext_menu:
            self.contextMenu.exec_(QCursor.pos())

    def emitAction(self, name):
        self.actionSignal.emit(name)

    def focus_cursor(self):
        cur_pos = self.mapFromGlobal(QCursor().pos())
        return self.transformPos(cur_pos)

    def offset_center(self):
        dx = self.width() - self.picture.width() * self.scale
        dy = self.height() - self.picture.height() * self.scale
        pos = QPointF(dx / 2, dy / 2)
        self.org = pos
        return pos

    _b_full_screen = False
    _old_parent = None
    _geometry = None

    def on_show_full_screen(self):
        if not self._b_full_screen:
            self.show_full_screen()
        else:
            self.cancel_full_screen()

    def show_full_screen(self):
        self._b_full_screen = True
        self._old_parent = self.parent()
        self._geometry = self.saveGeometry()
        self.setParent(None)
        self.showFullScreen()
        self.zoom_buttons[3].setIcon(newIcon("show_normal"))
        self.zoom_buttons[3].setToolTip("Show normal")

    def cancel_full_screen(self):
        self._b_full_screen = False
        self.setParent(self._old_parent)
        self.parent().setCentralWidget(self)
        self.zoom_buttons[3].setIcon(newIcon("full_screen"))
        self.zoom_buttons[3].setToolTip("Show full screen")

    def fit_window(self):
        if self.picture is None:
            return
        self.scale = self.scaleFitWindow()
        self.org = self.offset_center()

    def scaleFitWindow(self):
        e = 2.0
        w1 = self.width() - 2
        h1 = self.height() - 2
        a1 = w1 / h1
        w2 = self.picture.width()
        h2 = self.picture.height()
        a2 = w2 / h2
        return w1 / w2 if a2 >= a1 else h1 / h2

    def zoom_origin(self):
        self.scale = 1
        self.org = QPointF()

    def zoom_manual(self, s):
        self.scale *= s
        self.zoomSignal.emit(self.scale)
        return

    def zoom_focus_cursor(self, s):
        old_scale = self.scale
        p1 = self.current_pos
        self.scale *= s
        # focus cursor pos
        self.org -= p1 * self.scale - p1 * old_scale

    def zoom_by_wheel(self, s):
        self.zoom_focus_cursor(s)
        # self.repaint()
        self.zoomSignal.emit(self.scale)
        return

    def transformPos(self, pos):
        """
        convert main pos -> cv pos
        """
        return (QPointF(pos) - self.org) / self.scale

    def move_org(self, point):
        self.org += point

    def update_center(self, pos):
        pass

    def draw_rect(self, pos1, pos2):
        self.current_rect = QRectF(pos1, pos2)

    def shape_to_cvRect(self, shape):
        p1 = shape.points[0]
        p2 = shape.points[2]
        x, y = p1.x(), p1.y()
        x2, y2 = p2.x(), p2.y()
        #
        x = max(x, 0)
        y = max(y, 0)
        x2 = min(x2, self.picture.width())
        y2 = min(y2, self.picture.height())
        #
        w, h = int(x2 - x), int(y2 - y)
        x, y = int(x), int(y)
        #
        return (x, y, w, h)

    def editShape(self):
        if self.idSelected is not None:
            label = self.boxEditLabel.popUp(self.last_label, self.labels, bMove=True)
            if label:
                self[self.idSelected].label = label
                self.last_label = label
                self.append_new_label(label)

    def copyShape(self):
        if self.idSelected is not None:
            shape = self[self.idSelected].copy()
            self.shapes.append(shape)
            # self.dict_shapes[shape.label] = shape
            i = self.idSelected
            # self.releae_shape_selected(i)
            self.idSelected = i + 1

    def undo(self):
        if len(self.shapes) > 0:
            self.shapes.remove(self[-1])

    def deleteShape(self):
        if self.idSelected is not None:
            # self.emitAction("remove")
            shape = self[self.idSelected]
            self.deleteShapeSignal.emit(self.idSelected)
            self.shapes.remove(shape)
            # del(self.dict_shapes[shape.label])

            self.idVisible = self.idSelected = self.idCorner = None

    def delete_all(self):
        for i in range(len(self)):
            self.deleteShapeSignal.emit(len(self) - 1)
            self.shapes.remove(self.shapes[-1])
        self.idVisible = self.idSelected = self.idCorner = None

    def moveShape(self, i, v):
        if self.picture is None:
            return
        self[i].move(v)
        self.moveShapeSignal.emit(i)

    def append_new_label(self, label):
        if label not in self.labels:
            self.labels.append(label)
            self.labels = [lb.strip("\r\n") for lb in self.labels]
            string = "\n".join(self.labels)
            if os.path.exists(self.label_path):
                with open(self.label_path, "w") as ff:
                    ff.write(string)
                    ff.close()
        pass

    def newShape(self, r, label):
        labels = [s.label for s in self.shapes]
        if label in labels:
            QMessageBox.warning(self, "WARNING", "Shape already exists")
            return
        # n = len(self)
        # label = "Shape-%d"%n
        shape = Shape(label)
        ret, points = shape.get_points(r)
        if ret:
            shape.points = points
            self.shapes.append(shape)
            # self.dict_shapes[label] = shape
            self.newShapeSignal.emit(len(self) - 1)
            self.last_label = label
            self.append_new_label(label)
        return shape

    def format_shape(self, shape):
        label = shape.label
        r = self.shape_to_cvRect(shape)
        id = self.shapes.index(shape)
        return {"label": label, "box": r, "id": id}

    def pos_in_shape(self, pos, shape):
        pass

    def visibleShape(self, pos):
        n = len(self)
        ids_shape_contain_pos = []
        distances = []
        for i in range(n):
            self[i].visible = False
            d = self[i].dis_to(pos)
            if d > 0:
                ids_shape_contain_pos.append(i)
                distances.append(d)

        if len(distances) > 0:
            index = np.argmin(distances)
            self.idVisible = ids_shape_contain_pos[index]
            self[self.idVisible].visible = True
            # self.visi.emit(self.idSelected)
        else:
            self.idVisible = None
        return self.idVisible

    def selectedShape(self, pos):
        ids_shape_contain_pos = []
        distances = []

        s: Shape = None
        for i, s in enumerate(self.shapes):
            if not s.hide:
                s.selected = False
                d = s.dis_to(pos)
                if d > 0:
                    ids_shape_contain_pos.append(i)
                    distances.append(d)

        if len(distances) > 0:
            index = np.argmin(distances)
            self.idSelected = ids_shape_contain_pos[index]
            self[self.idSelected].selected = True
            self.selectedShapeSignal.emit(self.idSelected)
        else:
            self.idSelected = None

    def highlightCorner(self, pos, epsilon=10):
        if self.idSelected is None:
            return False
        try:
            i = self.idSelected
            return self[i].get_corner(pos, epsilon)
        except Exception as ex:
            print("{}".format(ex))
            return False

    def cancel_edit(self):
        self.edit = False
        self.drawing = False
        self.moving = False

    def cancel_selected(self):
        n = len(self)
        for i in range(n):
            self[i].selected = False
            self[i].corner = None
            self[i].visible = False
        self.idSelected = None

    def paintEvent(self, event):
        r: QRect = self.geometry()
        self.label_pos.setGeometry(0, r.height() - 30, r.width(), 30)

        w = 150
        self.tool_zoom.setGeometry((r.width() - w) // 2, r.y(), w, 30)

        if self.picture is None:
            return super(Canvas, self).paintEvent(event)

        p: QPainter = self.painter
        p.begin(self)
        lw = max(int(Shape.THICKNESS / (self.scale + 1e-3)), 1)
        p.setPen(QPen(QColor("green"), lw))
        p.translate(self.org)
        p.scale(self.scale, self.scale)

        if self.picture:
            p.drawPixmap(0, 0, self.picture)

        shape: Shape = None
        for shape in self.shapes:
            shape.paint(p, self.scale)

        if self.edit:
            # draw center
            pos = self.current_pos
            self.line1 = [QPointF(0, pos.y()), QPointF(self.picture.width(), pos.y())]
            self.line2 = [QPointF(pos.x(), 0), QPointF(pos.x(), self.picture.height())]
            p.drawLine(self.line1[0], self.line1[1])
            p.drawLine(self.line2[0], self.line2[1])

        if self.drawing:  # draw rect
            if self.current is not None:
                p.drawRect(self.current)

        self.update()
        p.end()

        return super().paintEvent(event)

    def wheelEvent(self, ev):
        if self.picture is None:
            return super(Canvas, self).wheelEvent(ev)
        delta = ev.angleDelta()
        h_delta = delta.x()
        v_delta = delta.y()
        mods = ev.modifiers()
        # if Qt.ControlModifier == int(mods) and v_delta:
        if v_delta:
            self.zoom_by_wheel(1 + v_delta / 120 * 0.2)
        # else:
        #     pos = QPointF(0.,v_delta/8.)
        #     self.move_org(pos)
        #     pass

        ev.accept()

    def mousePressEvent(self, ev):
        if self.picture is None:
            return super(Canvas, self).mousePressEvent(ev)
        # pos = self.transformPos(ev.pos())
        self.start_pos = self.transformPos(ev.pos())
        if ev.button() == Qt.MouseButton.LeftButton:
            if self.edit:
                if self.idSelected is not None:
                    self[self.idSelected].selected = False
                    self.idSelected = None
                self.drawing = True
            else:
                self.moving = True
                if not self.highlight:
                    self.selectedShape(self.start_pos)

    def mouseReleaseEvent(self, ev):
        if self.picture is None:
            return super(Canvas, self).mouseReleaseEvent(ev)
        # pos = self.transformPos(ev.pos())
        self.move_shape = False
        if ev.button() == Qt.MouseButton.LeftButton:
            if self.drawing:
                r = self.current
                if (
                    r is not None
                    and r.width() > Shape.MIN_WIDTH
                    and r.height() > Shape.MIN_WIDTH
                ):
                    label = self.boxEditLabel.popUp(
                        self.last_label, self.labels, bMove=False
                    )
                    if label:
                        self.newShape(r, label)
                self.current = None

            self.cancel_edit()

    def mouseMoveEvent(self, ev):
        if self.picture is None:
            return super(Canvas, self).mouseMoveEvent(ev)

        self.current_pos: QPointF = self.transformPos(ev.pos())

        image = self.picture.toImage()
        try:
            pos: QPoint = self.current_pos.toPoint()
            if (
                self.picture.width() > pos.x() >= 0
                and self.picture.height() > pos.y() >= 0
            ):
                pixel: QColor = image.pixelColor(pos)
                h, s, v, _ = pixel.getHsv()
                r, g, b, _ = pixel.getRgb()
                x, y = pos.x(), pos.y()
                self.text_pixel_color = (
                    "POS: [%d, %d], BGR: [%d, %d, %d], HSV: [%d, %d, %d]"
                    % (x, y, b, g, r, h, s, v)
                )
                self.label_pos.setText(self.text_pixel_color)
        except Exception as ex:
            pass

        self.mouseMoveSignal.emit(self.current_pos)
        if self.drawing:
            self.current = QRectF(self.start_pos, self.current_pos)
            self.drawShapeSignal.emit(self.current)
            # self.override_cursor(CURSOR_MOVE)
        if not self.moving:
            self.highlight = self.highlightCorner(self.current_pos, epsilon=40)
            if self.highlight:
                # self.override_cursor(CURSOR_DRAW)
                pass
        if self.moving:
            v = self.current_pos - self.start_pos
            index = self.idSelected
            s: Shape = None
            if index is not None and not self[index].lock:
                s = self[index]
                if self.highlight:
                    s.change(v)
                else:
                    s.move(v)
            else:
                self.move_org(v * self.scale)

            self.start_pos = self.transformPos(ev.pos())
            if self.idSelected is not None:
                self.changeShapeSignal.emit(self.idSelected)
            # self.override_cursor(CURSOR_MOVE)

        if (
            self.visibleShape(self.current_pos) is None
            and not self.highlight
            and not self.drawing
        ):
            self.restore_cursor()
        elif not self.highlight and not self.drawing and not self.moving:
            pass
            # self.override_cursor(CURSOR_GRAB)
        if self.edit:
            pass
            # self.restore_cursor()

    def currentCursor(self):
        cursor = QApplication.overrideCursor()
        if cursor is not None:
            cursor = cursor.shape()
        return cursor

    def overrideCursor(self, cursor):
        self._cursor = cursor
        if self.currentCursor() is None:
            QApplication.setOverrideCursor(cursor)
        else:
            QApplication.changeOverrideCursor(cursor)

    def resizeEvent(self, ev):
        if self.picture is None:
            return super(Canvas, self).resizeEvent(ev)
        self.fit_window()
        pass

    def keyPressEvent(self, ev):
        key = ev.key()
        step = 5
        # if key == Qt.Key.Key_1:
        #     self.parent.io_signal = not self.parent.io_signal
        if key == Qt.Key.Key_W:
            if self.benable_drawing:
                self.edit = True

        elif key == Qt.Key.Key_Escape:
            self.cancel_edit()
            self.cancel_selected()
            if self._b_full_screen:
                self.cancel_full_screen()

        elif key == Qt.Key.Key_Delete:
            self.deleteShape()

        elif key == Qt.Key.Key_Return:
            self.fit_window()

        elif key == Qt.Key.Key_Plus:
            s = 1.2
            self.zoom_focus_cursor(s)

        elif key == Qt.Key.Key_Minus:
            s = 0.8
            self.zoom_focus_cursor(s)

        i = self.idSelected
        if i is not None:
            if key == Qt.Key.Key_Right:
                v = QPointF(step, 0)
                self.moveShape(i, v)
            elif key == Qt.Key.Key_Left:
                v = QPointF(-step, 0)
                self.moveShape(i, v)
            elif key == Qt.Key.Key_Up:
                v = QPointF(0, -step)
                self.moveShape(i, v)
            elif key == Qt.Key.Key_Down:
                v = QPointF(0, step)
                self.moveShape(i, v)

        else:
            if self.picture is not None:
                step = min(self.picture.width() // 20, 10)
            else:
                step = 10

            if key == Qt.Key.Key_Right:
                v = QPointF(step, 0)
                self.move_org(v)
            elif key == Qt.Key.Key_Left:
                v = QPointF(-step, 0)
                self.move_org(v)
            elif key == Qt.Key.Key_Up:
                v = QPointF(0, -step)
                self.move_org(v)
            elif key == Qt.Key.Key_Down:
                v = QPointF(0, step)
                self.move_org(v)

    def load_pixmap(self, pixmap, fit=False):
        self.picture = pixmap
        if fit:
            self.fit_window()
        self.zoomSignal.emit(self.scale)
        self.repaint()

    def current_cursor(self):
        cursor = QApplication.overrideCursor()
        if cursor is not None:
            cursor = cursor.shape()
        return cursor

    def override_cursor(self, cursor):
        self._cursor = cursor
        if self.current_cursor() is None:
            QApplication.setOverrideCursor(cursor)
        else:
            QApplication.changeOverrideCursor(cursor)

    def restore_cursor(self):
        QApplication.restoreOverrideCursor()

    def __len__(self):
        return len(self.shapes)

    def __getitem__(self, key):
        # if isinstance(key,int):
        return self.shapes[key]

    # elif isinstance(key,str):
    #     return self.dict_shapes[key]

    def __setitem__(self, key, value):
        self.shapes[key] = value

    def clear_pixmap(self):
        self.picture = None

    def clear(self):
        self.shapes.clear()
        self.idSelected = None
        self.idVisible = None
        self.idCorner = None


class WindowCanvas(QMainWindow):
    def __init__(self, canvas=None, parent=None):
        super().__init__(parent=parent)
        self.setCentralWidget(canvas)
        self.setObjectName("WindowCanvas")

        self.setStyleSheet(
            """
            QMainWindow {
                border: 1px solid navy;  /* Dashed red border */
                border-radius: 5px;    /* Rounded corners */
                background-color: lightgray;  /* Optional: background color */
            }
        """
        )


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    # wd = QMainWindow()

    canvas = Canvas()
    canvas.load_pixmap(QPixmap(640, 480))

    # wd.setCentralWidget(canvas)
    canvas.showMaximized()

    sys.exit(app.exec_())
