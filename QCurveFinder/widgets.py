from PyQt5.QtWidgets import QLabel, QHBoxLayout, QVBoxLayout, QPushButton, QCheckBox, QLineEdit, QTextBrowser, \
    QSlider, QComboBox, QRadioButton, QSpinBox, QButtonGroup
from PyQt5.QtGui import QPixmap, QMouseEvent, QFont, QPainter, QPainterPath, QPen, QColor
from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QRect, QSize, QRectF, QSizeF

from typing import List, Tuple

from .constants import *


class QImage(QLabel):
    """ The class for the big image box """

    signal: pyqtSignal = pyqtSignal(int, int, Qt.MouseButton)
    zoom = 2
    radius = 60
    border: int = 3

    def __init__(self, image_path: str) -> None:
        """ Initialise the image of the graph """
        super().__init__()

        self.source: str = image_path  # Set the image

        self.clickEnabled: bool = False
        self.zoomEnabled: bool = False
        self.maskEnabled = False
        self.holding: bool = False
        self.button: Qt.MouseButton = Qt.MouseButton.NoButton
        self.setMouseTracking(True)

        self.setStyleSheet(f"border: {self.border}px solid gray;")  # Add borders

        self.base_pixmap: QPixmap = None
        self.contour_pixmap: QPixmap = None

        self.brush_radius: int = 5

    @property
    def maskEnabled(self) -> bool:
        """ Return if maskEnable is true or not """
        return self._maskEnabled

    @maskEnabled.setter
    def maskEnabled(self, a0: bool) -> None:
        """ Set the maskEnable attribute """
        self._maskEnabled = a0
        self.contour_pixmap = self.source.copy()

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
        self._source.save(src)  # Save the resized image
        self.setPixmap(self._source)  # Display the image
        self.base_pixmap = self._source.copy()

    def update_brush_radius(self, value: int) -> None:
        self.brush_radius = value

    def emit_signal(self, ev: QMouseEvent) -> None:
        self.signal.emit(ev.x() - self.border, ev.y() + self.border, self.button)

    def mousePressEvent(self, ev: QMouseEvent) -> None:
        """ Event when mouse is pressed on the image """
        if self.clickEnabled:
            self.holding = True
            self.button = ev.button()
            self.emit_signal(ev)

    def mouseMoveEvent(self, ev: QMouseEvent) -> None:
        """ Event when mouse is moved on the image """
        x, y = ev.x() - self.border, ev.y() + self.border

        if self.holding and self.maskEnabled:
            self.add_mask(x, y)
            self.emit_signal(ev)

        if self.zoomEnabled:
            self.add_zoom(x, y)

    def mouseReleaseEvent(self, ev: QMouseEvent) -> None:
        """ Event when mouse is released on the image """
        self.holding = False
        self.button = Qt.MouseButton.NoButton

    def add_mask(self, x: int, y: int) -> None:
        rectangle = QRect(QPoint(x - self.brush_radius, y - self.brush_radius),
                          2*self.brush_radius*QSize(1, 1))
        overlay_pixmap = self.base_pixmap.copy(rectangle)

        crosshair = QPainter(overlay_pixmap)
        if self.button == Qt.MouseButton.LeftButton:
            pen = QPen(QColor(255, 0, 0, 100), 3*self.brush_radius)
        elif self.button == Qt.MouseButton.RightButton:
            pen = QPen(QColor(255, 0, 0, 0), 3*self.brush_radius)
        else:
            pen = QPen(QColor(255, 0, 0, 80), 3*self.brush_radius)
        crosshair.setPen(pen)
        crosshair.drawPoint(overlay_pixmap.rect().center())
        crosshair.end()

        path = QPainterPath()
        rectangle = QRectF(QPoint(x - self.brush_radius, y - self.brush_radius),
                           2*self.brush_radius*QSizeF(1, 1))
        path.addEllipse(rectangle)

        painter = QPainter(self.contour_pixmap)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        painter.setClipPath(path, Qt.IntersectClip)
        painter.drawPixmap(QPoint(x - self.brush_radius, y - self.brush_radius), overlay_pixmap)
        painter.end()

        self.setPixmap(self.contour_pixmap)

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


class QCoord(QVBoxLayout):
    """ The class for the coordinate inputs box """

    pts_labels: Tuple[str] = ("X1", "X2", "Y1", "Y2")

    def __init__(self) -> None:
        """ Initialise the coordinate inputs """
        super().__init__()

        # Init the pts and them characteristics
        self.pts: List[Tuple[int, int]] = [(-1, -1)]*4

        # Create the widget
        self.x1_coord = self.QCoordBox(self.pts_labels[0])
        self.x2_coord = self.QCoordBox(self.pts_labels[1])
        self.y1_coord = self.QCoordBox(self.pts_labels[2])
        self.y2_coord = self.QCoordBox(self.pts_labels[3])

        # Create the layout
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
        self.textbox.setMarkdown("Select a graph to start.")
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

    options: Tuple[str] = ("Canny", "Global Tresholding", "Adaptive Mean Tresholding", "Adaptive Gausian Tresholding",
                           "Otsu's Tresholding", "Otsu's Tresholding + Gausian Blur")
    treshs: Tuple[Tuple[bool]] = ((True, True), (True, True), (False, False), (False, False),
                                  (False, False), (False, False), (False, False))
    tresh_ext: Tuple[Tuple[int]] = ((-1000, 1000, -1000, 1000), (0, 255, 0, 255), (0, 1, 0, 1), (0, 1, 0, 1),
                                    (0, 1, 0, 1), (0, 1, 0, 1), (0, 1, 0, 1))

    def __init__(self) -> None:
        """ Initialise the image options """
        super().__init__()

        # Set the radios
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
        self.combo: QComboBox = QComboBox()
        self.combo.addItems(self.options)
        self.combo.currentTextChanged.connect(self.combo_change)

        # Set the first labeled slider
        self.label1: QLabel = QLabel(text="Tresh. 1")
        self.slider1: QSlider = QSlider(Qt.Horizontal)
        self.slider1.setMinimum(-1000)
        self.slider1.setMaximum(1000)
        self.slider1.setTickInterval(100)
        self.slider1.setTickPosition(QSlider.TicksBelow)

        # Set the second labeled slider
        self.label2: QLabel = QLabel(text="Tresh. 2")
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
        self.addLayout(hb0)
        self.addLayout(hb1)
        self.addWidget(self.combo)
        self.addLayout(hb2)
        self.addLayout(hb3)
        self.addLayout(hb4)

    def combo_change(self, text) -> None:
        """ Method to change the slider values based on the combobox text """
        for (i, op) in enumerate(self.options):
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
