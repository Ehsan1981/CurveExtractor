from PyQt5.QtWidgets import QLabel, QHBoxLayout, QVBoxLayout, QPushButton, QCheckBox, QLineEdit, QTextBrowser, \
    QSlider, QComboBox, QRadioButton, QSpinBox, QButtonGroup
from PyQt5.QtGui import QPixmap, QMouseEvent, QFont, QPainter, QPainterPath, QPen, QColor, QTextDocument
from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QRect, QSize, QRectF, QSizeF

from typing import List, Tuple

from .constants import *


class QImage(QLabel):
    """ The class for the big image box """

    signal: pyqtSignal = pyqtSignal(int, int, Qt.MouseButton)
    zoom = 2
    radius = 60
    border: int = 3
    pts_colors: Tuple[QColor] = (QColor(204, 0, 0, 150), QColor(0, 153, 0, 150),
                                 QColor(0, 0, 153, 150), QColor(204, 204, 0, 150))
    pts_labels: Tuple[str] = ("X<sub>1</sub>", "X<sub>2</sub>", "Y<sub>1</sub>", "Y<sub>2</sub>")

    def __init__(self, image_path: str) -> None:
        """ Initialise the image of the graph """
        super().__init__()

        self.source: str = image_path  # Set the image

        self.clickEnabled: bool = False
        self.zoomEnabled: bool = False
        self.maskEnabled = False
        self.coordEnabled = False
        self.holding: bool = False
        self.button: Qt.MouseButton = Qt.MouseButton.NoButton
        self.setMouseTracking(True)

        self.setStyleSheet(f"border: {self.border}px solid gray;")  # Add borders

        self.base_pixmap: QPixmap = None
        self.updated_pixmap: QPixmap = None

        self.brush_radius: int = 5
        self.pts: List[QPoint, QPoint, QPoint, QPoint] = [None, None, None, None]

    def mousePressEvent(self, ev: QMouseEvent) -> None:
        """ Event when mouse is pressed on the image """
        x, y = self.get_xy_from_event(ev)

        if self.clickEnabled:
            self.holding = True
            self.button = ev.button()
            if self.maskEnabled:
                self.add_mask(x, y)
            if self.coordEnabled:
                self.add_coord(x, y)
            self.emit_signal(ev)

    def mouseMoveEvent(self, ev: QMouseEvent) -> None:
        """ Event when mouse is moved on the image """
        x, y = self.get_xy_from_event(ev)

        if self.maskEnabled:
            self.add_mask(x, y)
            if self.holding:
                self.emit_signal(ev)

        if self.zoomEnabled:
            self.add_zoom(x, y)

    def mouseReleaseEvent(self, ev: QMouseEvent) -> None:
        """ Event when mouse is released on the image """
        self.holding = False
        self.button = Qt.MouseButton.NoButton

    def add_coord(self, x: int, y: int) -> None:
        if self.button == Qt.MouseButton.LeftButton:
            painter = QPainter(self.base_pixmap)

            # Set the pen for the point
            pen = QPen(self.pts_colors[self.num_printed_coord], 15)
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            painter.drawPoint(QPoint(x, y))

            # Draw text
            text = QTextDocument()
            text.setHtml(self.pts_labels[self.num_printed_coord])
            font = text.defaultFont()
            font.setPointSize(14)
            text.setDefaultFont(font)
            painter.translate(QPoint(x, y))
            text.drawContents(painter)
            painter.end()

            self.setPixmap(self.base_pixmap)

            self.pts[self.num_printed_coord] = QPoint(x, y)

            self.num_printed_coord += 1

    def add_mask(self, x: int, y: int) -> None:
        rectangle = QRect(QPoint(x - self.brush_radius, y - self.brush_radius),
                          2*self.brush_radius*QSize(1, 1))
        overlay_pixmap = self.base_pixmap.copy(rectangle)
        crosshair_pixmap = self.base_pixmap.copy(rectangle)

        # If holding, add the mask
        if self.holding and self.button == Qt.MouseButton.LeftButton or self.button == Qt.MouseButton.RightButton:
            brush_painter = QPainter(overlay_pixmap)
            if self.button == Qt.MouseButton.LeftButton:
                pen = QPen(QColor(255, 0, 0, 100), 3*self.brush_radius)
            elif self.button == Qt.MouseButton.RightButton:
                pen = QPen(QColor(255, 0, 0, 0), 3*self.brush_radius)
            brush_painter.setPen(pen)
            brush_painter.drawPoint(overlay_pixmap.rect().center())
            brush_painter.end()

        # Add a contour to visualize where you paint
        crosshair_painter = QPainter(crosshair_pixmap)
        if self.holding and self.button == Qt.MouseButton.LeftButton:
            pen = QPen(QColor(255, 0, 0, 100), 3*self.brush_radius)
            crosshair_painter.setPen(pen)
            crosshair_painter.drawPoint(overlay_pixmap.rect().center())
        crosshair_painter.setPen(QPen(QColor(255, 255, 255, 80), 3))
        crosshair_painter.drawEllipse(crosshair_pixmap.rect())
        crosshair_painter.end()

        path = QPainterPath()
        rectangle = QRectF(QPoint(x - self.brush_radius, y - self.brush_radius), 2*self.brush_radius*QSizeF(1, 1))
        path.addEllipse(rectangle)

        # If holding, add the mask to the updated pixmap
        if self.holding:
            mask_painter = QPainter(self.updated_pixmap)
            mask_painter.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
            mask_painter.setClipPath(path, Qt.IntersectClip)
            mask_painter.drawPixmap(QPoint(x - self.brush_radius, y - self.brush_radius), overlay_pixmap)
            mask_painter.end()

        # Add the crosshair to the pixmap
        rendered_pixmap = self.updated_pixmap.copy()
        final_crosshair_painter = QPainter(rendered_pixmap)
        final_crosshair_painter.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        final_crosshair_painter.setClipPath(path, Qt.IntersectClip)
        final_crosshair_painter.drawPixmap(QPoint(x - self.brush_radius, y - self.brush_radius), crosshair_pixmap)
        final_crosshair_painter.end()

        self.setPixmap(rendered_pixmap)

    def add_zoom(self, x: int, y: int) -> None:
        base_pixmap = self.base_pixmap.copy()
        rectangle = QRect(QPoint(x - self.radius/2, y - self.radius/2), self.radius*QSize(1, 1))
        overlay_pixmap = base_pixmap.copy(rectangle).scaledToWidth(self.zoom*self.radius, Qt.SmoothTransformation)

        crosshair = QPainter(overlay_pixmap)
        crosshair.setPen(QPen(Qt.black, 3))
        crosshair.drawPoint(overlay_pixmap.rect().center())
        crosshair.drawEllipse(overlay_pixmap.rect())
        crosshair.end()

        rectangle_zoomed = QRectF(QPoint(x, y), self.zoom*self.radius*QSizeF(1, 1))
        path = QPainterPath()
        path.addEllipse(rectangle_zoomed)

        painter = QPainter(base_pixmap)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        painter.setClipPath(path, Qt.IntersectClip)
        painter.drawPixmap(QPoint(x, y), overlay_pixmap)
        painter.end()

        self.setPixmap(base_pixmap)

    def draw_points(self, pts: Tuple[Tuple[float, float], ...]) -> None:
        for i, pt in enumerate(pts):
            x, y = pt
            self.pts[i] = QPoint(int(x), int(y))

        for pt, color, label in zip(self.pts, self.pts_colors, self.pts_labels):
            painter = QPainter(self.base_pixmap)
            # Set the pen for the point
            pen = QPen(color, 15)
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            painter.drawPoint(pt)

            # Draw text
            text = QTextDocument()
            text.setHtml(label)
            font = text.defaultFont()
            font.setPointSize(14)
            text.setDefaultFont(font)
            painter.translate(pt)
            text.drawContents(painter)

            painter.end()

        self.setPixmap(self.base_pixmap)

    def update_brush_radius(self, value: int) -> None:
        self.brush_radius = value

    def emit_signal(self, ev: QMouseEvent) -> None:
        x, y = self.get_xy_from_event(ev)
        self.signal.emit(x, y, self.button)

    def get_xy_from_event(self, ev: QMouseEvent) -> Tuple[int, int]:
        return ev.x() - self.border, ev.y() + self.border

    @property
    def maskEnabled(self) -> bool:
        """ Return if maskEnable is true or not """
        return self._maskEnabled

    @maskEnabled.setter
    def maskEnabled(self, a0: bool) -> None:
        """ Set the maskEnable attribute """
        self._maskEnabled = a0
        self.updated_pixmap = self.source.copy()

    @property
    def coordEnabled(self) -> bool:
        """ Return if coordEnable is true or not """
        return self._coordEnabled

    @coordEnabled.setter
    def coordEnabled(self, a0: bool) -> None:
        """ Set the coordEnable attribute """
        self._coordEnabled = a0
        self.updated_pixmap = self.source.copy()
        self.num_printed_coord = 0

    @property
    def source(self) -> QPixmap:
        """ Return the image """
        return self._source

    @source.setter
    def source(self, src: str) -> None:
        """ Set the image and resize it to fit in the box """
        self.img_path = src  # Save the path
        new_img = QPixmap(src)  # Load the image
        self._source = new_img.scaled(MAX_IMG_W, MAX_IMG_H, Qt.KeepAspectRatio)  # Save the rescaled source image
        self.image_size = (self._source.height(), self._source.width())
        self._source.save(src)  # Save the resized image
        self.setPixmap(self._source)  # Display the image
        self.base_pixmap = self._source.copy()


class QCoord(QVBoxLayout):
    """ The class for the coordinate inputs box """

    pts_labels: Tuple[str] = ("X1", "X2", "Y1", "Y2")

    def __init__(self) -> None:
        """ Initialise the coordinate inputs """
        super().__init__()

        # Init the pts and them characteristics
        self.pts: List[Tuple[int, int]] = [(-1, -1)]*4

        # Create the widget
        self.label = QLabel(text="Coordinates :")
        self.label.setFont(QFont("Helvetica", 10, QFont.Bold))

        self.x1_coord = self.QCoordBox(self.pts_labels[0])
        self.x2_coord = self.QCoordBox(self.pts_labels[1])
        self.y1_coord = self.QCoordBox(self.pts_labels[2])
        self.y2_coord = self.QCoordBox(self.pts_labels[3])

        # Create the layout
        self.addWidget(self.label)
        self.addLayout(self.x1_coord)
        self.addLayout(self.x2_coord)
        self.addLayout(self.y1_coord)
        self.addLayout(self.y2_coord)

        # Initialise the values
        self.initValues()

    def initValues(self) -> None:
        """ Initialise the coordinate statuses """
        self.x1_done = False
        self.x2_done = False
        self.y1_done = False
        self.y2_done = False

    @property
    def x1_done(self) -> bool:
        """ Return if x1 is done """
        return self._x1_done

    @x1_done.setter
    def x1_done(self, status: bool) -> None:
        """ Set x1 and check the box """
        self._x1_done = status
        self.x1_coord.check.setChecked(status)
        if not status:
            self.pts[0] = (-1, -1)

    @property
    def x2_done(self) -> bool:
        """ Return if x2 is done """
        return self._x2_done

    @x2_done.setter
    def x2_done(self, status: bool) -> None:
        """ Set x2 and check the box """
        self._x2_done = status
        self.x2_coord.check.setChecked(status)
        if not status:
            self.pts[1] = (-1, -1)

    @property
    def y1_done(self) -> bool:
        """ Return if y1 is done """
        return self._y1_done

    @y1_done.setter
    def y1_done(self, status: bool) -> None:
        """ Set y1 and check the box """
        self._y1_done = status
        self.y1_coord.check.setChecked(status)
        if not status:
            self.pts[2] = (-1, -1)

    @property
    def y2_done(self) -> bool:
        """ Return if y2 is done """
        return self._y2_done

    @y2_done.setter
    def y2_done(self, status: bool) -> None:
        """ Set y2 and check the box """
        self._y2_done = status
        self.y2_coord.check.setChecked(status)
        if not status:
            self.pts[3] = (-1, -1)

    class QCoordBox(QHBoxLayout):
        """ Subclass for the a single coordinate box """

        def __init__(self, coord_label: str) -> None:
            """ Initialise the single coordinate input box """
            super().__init__()

            # Set the widgets
            self.coord_label: str = coord_label
            self.label: QLabel = QLabel(text=f"{coord_label} :")
            self.line: QLineEdit = QLineEdit()
            self.line.setPlaceholderText(f"Enter coord. for {coord_label}...")
            self.check: QCheckBox = QCheckBox(text=f"{coord_label} placed")
            self.check.setEnabled(False)

            # Create the layout
            self.addWidget(self.label)
            self.addWidget(self.line)
            self.addWidget(self.check)


class QInstructBox(QVBoxLayout):
    """ Class for the instruction box widget """

    def __init__(self) -> None:
        """ Initialise the Instruction box """
        super().__init__()

        # Create the  widgets
        self.label = QLabel(text="Instructions :")
        self.label.setFont(QFont("Helvetica", 14, QFont.Bold))

        self.combo: QComboBox = QComboBox()
        self.combo.addItems(COPY_OPTIONS_TEXT)
        self.but_copy: QPushButton = QPushButton(text="Copy")

        self.textbox: QTextBrowser = QTextBrowser()
        self.textbox.setFont(QFont("Calibri", 12, QFont.Bold))

        # Create the layout
        hbox = QHBoxLayout()
        hbox.addWidget(self.combo, stretch=1)
        hbox.addWidget(self.but_copy)

        vbox = QVBoxLayout()
        vbox.addWidget(self.label)
        vbox.addWidget(self.textbox)
        vbox.addLayout(hbox)

        self.addLayout(vbox)

    def setEnabled(self, a0: bool) -> None:
        """ Method to enable or disable the copy button and combobox """
        self.combo.setEnabled(a0)
        self.but_copy.setEnabled(a0)


class QImageOptions(QVBoxLayout):
    """ Class to create the images options widget """

    treshs: Tuple[Tuple[bool]] = ((True, True), (True, True), (False, False), (False, False),
                                  (False, False), (False, False), (False, False))
    tresh_ext: Tuple[Tuple[int]] = ((-1000, 1000, -1000, 1000), (0, 255, 0, 255), (0, 1, 0, 1), (0, 1, 0, 1),
                                    (0, 1, 0, 1), (0, 1, 0, 1), (0, 1, 0, 1))

    def __init__(self) -> None:
        """ Initialise the image options """
        super().__init__()

        # Set the radios
        self.label = QLabel(text="Image/graph options :")
        self.label.setFont(QFont("Helvetica", 10, QFont.Bold))

        self.label0: QLabel = QLabel(text="Wanted equation :")
        self.bg_formula: QButtonGroup = QButtonGroup()
        self.y_from_x: QRadioButton = QRadioButton(text="y = f(x)")
        self.y_from_x.setChecked(True)
        self.x_from_y: QRadioButton = QRadioButton(text="x = f(y)")
        self.bg_formula.addButton(self.y_from_x)
        self.bg_formula.addButton(self.x_from_y)

        self.labelx: QLabel = QLabel(text="X-axis is :")
        self.bg_x: QButtonGroup = QButtonGroup()
        self.x_lin: QRadioButton = QRadioButton(text="Lin.")
        self.x_lin.setChecked(True)
        self.x_log: QRadioButton = QRadioButton(text="Log.")
        self.bg_x.addButton(self.x_lin)
        self.bg_x.addButton(self.x_log)

        self.labely: QLabel = QLabel(text="Y-axis is :")
        self.bg_y: QButtonGroup = QButtonGroup()
        self.y_lin: QRadioButton = QRadioButton(text="Lin.")
        self.y_lin.setChecked(True)
        self.y_log: QRadioButton = QRadioButton(text="Log.")
        self.bg_y.addButton(self.y_lin)
        self.bg_y.addButton(self.y_log)

        # Set the combobox
        self.label_combo: QLabel = QLabel(text="Contour :")
        self.combo: QComboBox = QComboBox()
        self.combo.addItems(CONTOUR_OPTIONS_TEXT)
        self.combo.currentTextChanged.connect(self.combo_change)

        # Set the first labeled slider
        self.label1: QLabel = QLabel(text="Thresh. 1")
        self.slider1: QSlider = QSlider(Qt.Horizontal)
        self.slider1.setMinimum(-1000)
        self.slider1.setMaximum(1000)
        self.slider1.setTickInterval(100)
        self.slider1.setTickPosition(QSlider.TicksBelow)

        # Set the second labeled slider
        self.label2: QLabel = QLabel(text="Thresh. 2")
        self.slider2: QSlider = QSlider(Qt.Horizontal)
        self.slider2.setMinimum(-1000)
        self.slider2.setMaximum(1000)
        self.slider2.setTickInterval(100)
        self.slider2.setTickPosition(QSlider.TicksBelow)

        # Set the brush/order spinbox
        self.label3: QLabel = QLabel()
        self.spinbox: QSpinBox = QSpinBox()
        self.spinbox.setMinimum(0)
        self.is_brush: bool = True

        # Create the layout
        hb0 = QHBoxLayout()
        hb0.addWidget(self.label0)
        hb0.addStretch(1)
        hb0.addWidget(self.y_from_x)
        hb0.addWidget(self.x_from_y)

        hb1 = QHBoxLayout()
        hb1.addWidget(self.labelx)
        hb1.addWidget(self.x_lin)
        hb1.addWidget(self.x_log)
        hb1.addStretch(1)
        hb1.addWidget(self.labely)
        hb1.addWidget(self.y_lin)
        hb1.addWidget(self.y_log)

        hbcombo = QHBoxLayout()
        hbcombo.addWidget(self.label_combo)
        hbcombo.addWidget(self.combo, stretch=4)

        hb2 = QHBoxLayout()
        hb2.addWidget(self.label1)
        hb2.addWidget(self.slider1)

        hb3 = QHBoxLayout()
        hb3.addWidget(self.label2)
        hb3.addWidget(self.slider2)

        hb4 = QHBoxLayout()
        hb4.addWidget(self.label3)
        hb4.addWidget(self.spinbox)

        # Set the final layout
        self.addWidget(self.label)
        self.addLayout(hb0)
        self.addLayout(hb1)
        self.addLayout(hbcombo)
        self.addLayout(hb2)
        self.addLayout(hb3)
        self.addLayout(hb4)

    def combo_change(self, text) -> None:
        """ Method to change the slider values based on the combobox text """
        for (i, op) in enumerate(CONTOUR_OPTIONS_TEXT):
            if text == op:
                self.slider1.setEnabled(self.treshs[i][0])
                self.slider1.setValue(0)
                self.slider1.setMinimum(self.tresh_ext[i][0])
                self.slider1.setMaximum(self.tresh_ext[i][1])
                self.slider2.setEnabled(self.treshs[i][1])
                self.slider2.setValue(0)
                self.slider2.setMinimum(self.tresh_ext[i][2])
                self.slider2.setMaximum(self.tresh_ext[i][3])

    def setEnabled(self, a0: bool) -> None:
        """ Method to disable or enable the combobox and sliders """
        self.combo.setEnabled(a0)
        self.slider1.setEnabled(a0)
        self.slider2.setEnabled(a0)

    @property
    def is_brush(self):
        return self._is_brush

    @is_brush.setter
    def is_brush(self, status: bool):
        self._is_brush = status
        if status:
            self.label3.setText("Brush size :")
            self.spinbox.setMaximum(50)
            self.spinbox.setSingleStep(5)
            self.spinbox.setValue(5)
        else:
            self.label3.setText("Fit order :")
            self.spinbox.setMaximum(15)
            self.spinbox.setSingleStep(1)
            self.spinbox.setValue(5)
