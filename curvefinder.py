# This Python file uses the following encoding: utf-8
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QHBoxLayout, QVBoxLayout, QPushButton, QFileDialog,\
    QCheckBox, QLineEdit, QMessageBox, QTextEdit, QSlider, QComboBox, QRadioButton, QSpinBox, QButtonGroup
from PyQt5.QtGui import QPixmap, QMouseEvent, QFont, QIcon
from PyQt5.QtCore import Qt, pyqtSignal

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from random import seed, randrange
from typing import List, Tuple
from shutil import rmtree
import numpy as np
import math as mt
import cv2
import sys
import os


VER = "2.2"
AUTHOR = "Bruno-Pier Busque"

MAX_IMG_W = 1100
MAX_IMG_H = 625

DATA_PATH = "data/"
ICON_PATH = DATA_PATH + "icon.png"
PH_IMAGE_PATH = DATA_PATH + "placeholder.png"
TEMP_PATH = DATA_PATH + "temp/"
ORIG_IMG = TEMP_PATH + "original_img.png"
COOR_IMG = TEMP_PATH + "coordinate_img.png"
ROTA_IMG = TEMP_PATH + "rotated_img.png"
CONT_IMG = TEMP_PATH + "contoured_img.png"
CTMK_IMG = TEMP_PATH + "masked_contoured_img.png"
SELE_IMG = TEMP_PATH + "selected_img.png"
PLOT_IMG = TEMP_PATH + "plotted_img.png"


class QImage(QLabel):
    """ The class for the big image box """

    signal: pyqtSignal = pyqtSignal(int, int)
    holdEnabled: bool = False
    holding: bool = False

    def __init__(self, image_path: str) -> None:
        """ Initialise the image of the graph """
        super().__init__()
        self.setStyleSheet("border: 3px solid gray;")  # Add borders
        self.source: str = image_path  # Set the image

    @property
    def source(self) -> QPixmap:
        """ Return the image """
        return self._source

    @source.setter
    def source(self, src: str) -> None:
        """ Set the image and resize it to fit in the box """
        self.img_path = src  # Save the path
        new_img = QPixmap(src)  # Load the image
        self.original_image_size = (new_img.height(), new_img.width())  # Save the original image size
        self._source = new_img.scaled(MAX_IMG_W, MAX_IMG_H, Qt.KeepAspectRatio)  # Save the rescaled source image
        self.new_image_size = (self._source.height(), self._source.width())  # Save the rescaled image size
        self.setPixmap(self._source)  # Display the image

    def mousePressEvent(self, ev: QMouseEvent) -> None:
        """ Event when mouse is pressed on the image """
        self.holding = True
        x, y = [self.original_image_size[i]*(ev.x(), ev.y())[i]/self.new_image_size[i] for i in (0, 1)]
        self.signal.emit(x, y)

    def mouseMoveEvent(self, ev: QMouseEvent) -> None:
        """ Event when mouse is moved on the image """
        if self.holding and self.holdEnabled:
            x, y = [self.original_image_size[i] * (ev.x(), ev.y())[i] / self.new_image_size[i] for i in (0, 1)]
            self.signal.emit(x, y)

    def mouseReleaseEvent(self, ev: QMouseEvent) -> None:
        """ Event when mouse is released on the image """
        self.holding = False


class QCoord(QVBoxLayout):
    """ The class for the coordinate inputs box """

    pts_labels: List[str] = ["X1", "X2", "Y1", "Y2"]
    pts: List[Tuple[int]] = [(-1, -1)]*4

    def __init__(self) -> None:
        """ Initialise the coordinate inputs """
        super().__init__()

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
        # self.but_next.setEnabled(status)
        if not status:
            self.pts[3] = (-1, -1)
        # else:
        #    self.instruct.setMarkdown("Enter the graph coordinates for each points.\n"
        #                              "\n"
        #                              "Press `Next` if your satisfied with your points.\n"
        #                              "Press `Restart` if not.")

    class QCoordBox(QHBoxLayout):
        """ Subclass for the a single coordinate box """

        def __init__(self, coord_label: str) -> None:
            """ Initialise the single coordinate input box """
            super().__init__()

            # Set the widgets
            self.coord_label: str = coord_label
            self.label: QLabel = QLabel(text="{0} :".format(coord_label))
            self.line: QLineEdit = QLineEdit()
            self.line.setPlaceholderText("Enter coord. for {0}...".format(coord_label))
            self.check: QCheckBox = QCheckBox(text="{0} placed".format(coord_label))
            self.check.setEnabled(False)

            # Create the layout
            self.addWidget(self.label)
            self.addWidget(self.line)
            self.addWidget(self.check)


class QInstructBox(QVBoxLayout):
    """ Class for the instruction box widget """

    options: List[str] = ["Copy Formula - Matlab", "Copy Formula - Python", "Copy Formula - Markdown",
                          "Copy Points - Matlab", "Copy Points - Python", "Copy Points - NumPy", "Copy Points - CSV",
                          "Copy Coeff. - Matlab", "Copy Coeff. - Python", "Copy Coeff. - NumPy", "Copy Poly1D - NumPy"]

    def __init__(self) -> None:
        """ Initialise the Instruction box """
        super().__init__()

        # Create the  widgets
        self.label = QLabel(text="Instructions :")
        self.label.setFont(QFont("Helvetica", 14, QFont.Bold))

        self.combo: QComboBox = QComboBox()
        self.combo.addItems(self.options)
        self.but_copy: QPushButton = QPushButton(text="Copy")

        self.textbox: QTextEdit = QTextEdit()
        self.textbox.setMarkdown("Select a graph to start.")
        self.textbox.setEnabled(False)
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

    options: List[str] = ["Canny", "Global Tresholding", "Adaptive Mean Tresholding", "Adaptive Gausian Tresholding",
                          "Otsu's Tresholding", "Otsu's Tresholding + Gausian Blur"]
    treshs: List[Tuple[bool]] = [(True, True), (True, True), (False, False), (False, False),
                                 (False, False), (False, False), (False, False)]
    tresh_ext: List[Tuple[int]] = [(-1000, 1000, -1000, 1000), (0, 255, 0, 255), (0, 1, 0, 1), (0, 1, 0, 1),
                                   (0, 1, 0, 1), (0, 1, 0, 1), (0, 1, 0, 1)]

    def __init__(self) -> None:
        """ Initialise the image options """
        super().__init__()

        # Set the radios
        self.label0: QLabel = QLabel(text="Wanted formula :")
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


class CurveFinder(QWidget):
    """ The application in itself """

    img_src: str = PH_IMAGE_PATH
    coord: np.ndarray = np.zeros(4, dtype=float)
    pts_colors: List[Tuple[int]] = [(204, 0, 0), (0, 153, 0), (0, 0, 153), (204, 204, 0)]
    pts_labels: List[str] = ["X1", "X2", "Y1", "Y2"]
    pts_final_p: List[np.ndarray] = []
    pts_final_r: List[np.ndarray] = []
    pts_eval_r: List[np.ndarray] = []
    x_axis_curve: dict = {}
    y_axis_curve: dict = {}
    Xpr: float = 0.0
    Ypr: float = 0.0
    coef: tuple = []
    order: int = 5
    var: str = "x"
    islog: List[bool] = [False, False]
    mask: np.ndarray = None

    def __init__(self) -> None:
        """ Initialise the app """
        QWidget.__init__(self)
        self.setWindowTitle("CurveFinder v{0}".format(VER))
        self.setWindowIcon(QIcon(ICON_PATH))
        self.setFixedWidth(1500)
        self.setFixedHeight(750)

        # Create a temporary and data folder
        if not os.path.exists(DATA_PATH):
            os.mkdir(DATA_PATH)
        if not os.path.exists(TEMP_PATH):
            os.mkdir(TEMP_PATH)

        # Create widgets
        title_label = QLabel(text="CurveFinder v{0}".format(VER))
        title_label.setFont(QFont("Helvetica", 20, QFont.Bold))
        author_label = QLabel(text="by {0}".format(AUTHOR))
        author_label.setFont(QFont("Calibri", 10))
        self.img: QImage = QImage(PH_IMAGE_PATH)
        self.instruct: QInstructBox = QInstructBox()
        self.img_op: QImageOptions = QImageOptions()
        self.coord_prompt: QCoord = QCoord()
        self.but_browse: QPushButton = QPushButton(text="Select an image")
        self.but_start: QPushButton = QPushButton(text="Start")
        self.but_next: QPushButton = QPushButton(text="Next")

        # Bind the signals
        self.img.signal.connect(self.add_position)
        self.but_browse.clicked.connect(self.browse_for_image)
        self.but_start.clicked.connect(self.start)
        self.but_next.clicked.connect(self.next)
        self.instruct.but_copy.clicked.connect(self.copy_text)
        self.img_op.combo.currentTextChanged.connect(self.update_image)
        self.img_op.slider1.sliderMoved.connect(self.update_image)
        self.img_op.slider2.sliderMoved.connect(self.update_image)
        self.img_op.spinbox.valueChanged.connect(self.set_formula)
        self.img_op.y_from_x.toggled.connect(self.set_formula)
        self.img_op.x_from_y.toggled.connect(self.set_formula)
        self.img_op.x_lin.toggled.connect(self.update_lin_log)
        self.img_op.x_log.toggled.connect(self.update_lin_log)
        self.img_op.y_lin.toggled.connect(self.update_lin_log)
        self.img_op.y_log.toggled.connect(self.update_lin_log)

        # Create the layout
        options = QVBoxLayout()
        options.addLayout(self.instruct)
        options.addLayout(self.img_op)
        options.addLayout(self.coord_prompt)
        options.addWidget(self.but_browse)
        but_lay = QHBoxLayout()
        but_lay.addWidget(self.but_start)
        but_lay.addWidget(self.but_next)
        options.addLayout(but_lay)

        vbox = QVBoxLayout()
        vbox.addWidget(title_label)
        vbox.addWidget(author_label)
        vbox.addWidget(self.img, alignment=Qt.AlignCenter, stretch=4)

        hbox = QHBoxLayout()
        hbox.addLayout(vbox, stretch=4)
        hbox.addLayout(options, stretch=1)

        self.setLayout(hbox)

        # Set application state
        self.app_state = 0

        self.show()

    def __del__(self) -> None:
        """ Remove the temporary folder """
        rmtree(TEMP_PATH)

    def browse_for_image(self) -> None:
        """ Method to select an image """
        self.img_src = str(QFileDialog().getOpenFileName(filter="Images (*.png *.bmp *.jpg)")[0])
        if self.img_src != "":
            self.img.source = self.img_src
            self.app_state = 0  # Return to initial state

    def start(self) -> None:
        """ Method for the Start button """
        cv2.imwrite(ORIG_IMG, cv2.imread(self.img_src))
        self.app_state = 1

    def next(self) -> None:
        """ Method for the next button. It changes the state of the app """
        if self.app_state == 2 and self.verify_coord():
            self.app_state = 3

        elif self.app_state == 3:
            self.app_state = 4

        elif self.app_state == 4:
            self.app_state = 5

        elif self.app_state == 5:
            self.app_state = 6

        elif self.app_state == 6:
            self.app_state = 5

    def verify_coord(self) -> bool:
        """ Method to verify if the coordinates are entered in the input boxes """
        good_coord = True
        for (i, coord) in enumerate([self.coord_prompt.x1_coord, self.coord_prompt.x2_coord,
                                     self.coord_prompt.y1_coord, self.coord_prompt.y2_coord]):
            try:
                self.coord[i] = float(coord.line.text())
            except ValueError:
                msgBox = QMessageBox()
                msgBox.setIcon(QMessageBox.Warning)
                msgBox.setText("Coordinates of {0} must be an number!".format(coord.coord_label))
                msgBox.setWindowTitle("Warning")
                msgBox.setStandardButtons(QMessageBox.Ok)
                msgBox.exec()
                good_coord = False
                break

        return good_coord

    def add_position(self, x: int, y: int) -> None:
        """ Method used when clicking with the mouse on the image """
        if self.app_state == 1:
            if not self.coord_prompt.x1_done:
                self.draw_points(x1=(x, y))
                self.coord_prompt.x1_done = True
            elif not self.coord_prompt.x2_done:
                self.draw_points(x2=(x, y))
                self.coord_prompt.x2_done = True
            elif not self.coord_prompt.y1_done:
                self.draw_points(y1=(x, y))
                self.coord_prompt.y1_done = True
            elif not self.coord_prompt.y2_done:
                self.draw_points(y2=(x, y))
                self.coord_prompt.y2_done = True
                self.app_state = 2

        elif self.app_state == 4:
            self.draw_mask(x, y)

    def draw_points(self, x1: tuple = None, x2: tuple = None, y1: tuple = None, y2: tuple = None) -> None:
        """ Method to draw the point on th image """
        if x1 is not None:
            self.coord_prompt.pts[0] = x1
        if x2 is not None:
            self.coord_prompt.pts[1] = x2
        if y1 is not None:
            self.coord_prompt.pts[2] = y1
        if y2 is not None:
            self.coord_prompt.pts[3] = y2

        img = cv2.imread(ORIG_IMG)

        for (i, pt) in enumerate(self.coord_prompt.pts):
            if sum(pt) != -2:
                rad = int(self.img.original_image_size[0]/100)
                cv2.circle(img, pt, rad, self.pts_colors[i], -1)
                cv2.putText(img, self.pts_labels[i], pt, cv2.FONT_HERSHEY_SIMPLEX, int(rad/3), self.pts_colors[i], rad)

        cv2.imwrite(COOR_IMG, img)
        self.img.source = COOR_IMG

    def draw_mask(self, x: int, y: int) -> None:
        """ Method to draw the brush on the image """
        alpha = 0.3
        radius = self.img_op.spinbox.value()
        img = cv2.imread(CONT_IMG)
        new_img = img.copy()
        cv2.circle(new_img, (x, y), radius, (0, 0, 255), -1)
        cv2.addWeighted(new_img, alpha, img, 1 - alpha, 0, img)
        cv2.circle(self.mask, (x, y), radius, 255, -1)
        cv2.imwrite(CONT_IMG, img)
        self.img.source = CONT_IMG

    def resize_and_rotate(self) -> None:
        """
        Method to rotate the image after the coordinate are confirmed.
        """
        X1_x = self.x_axis_curve["X1"] = self.coord_prompt.pts[0][0]
        X1_y = self.x_axis_curve["Y1"] = self.coord_prompt.pts[0][1]
        X2_x = self.x_axis_curve["X2"] = self.coord_prompt.pts[1][0]
        X2_y = self.x_axis_curve["Y2"] = self.coord_prompt.pts[1][1]
        Y1_x = self.y_axis_curve["X1"] = self.coord_prompt.pts[2][0]
        Y1_y = self.y_axis_curve["Y1"] = self.coord_prompt.pts[2][1]
        Y2_x = self.y_axis_curve["X2"] = self.coord_prompt.pts[3][0]
        Y2_y = self.y_axis_curve["Y2"] = self.coord_prompt.pts[3][1]

        Ax = self.x_axis_curve["A"] = (X2_y - X1_y)/(X2_x - X1_x)
        Y_dx = Y2_x - Y1_x
        if Y_dx == 0:
            self.y_axis_curve["A"] = mt.inf
            X0 = Y1_x
        else:
            Ay = self.y_axis_curve["A"] = (Y2_y - Y1_y)/Y_dx
            X0 = (Ay*Y1_x - Ax*X1_x + X1_y - Y1_y)/(Ay - Ax)

        Y0 = Ax*(X0 - X1_x) + X1_y
        angle_x = mt.atan((X2_y - X1_y)/(X2_x - X1_x))
        angle_y = mt.atan(-(Y2_x - Y1_x)/(Y2_y - Y1_y))
        angle = (angle_x + angle_y)/2

        R = cv2.getRotationMatrix2D((X0, Y0), 180*angle/mt.pi, 1)
        img = cv2.imread(self.img_src)
        img = cv2.warpAffine(img, R, img.shape[1::-1], flags=cv2.INTER_LINEAR)

        cv2.imwrite(ROTA_IMG, img)
        self.img.source = ROTA_IMG

        self.x_axis_curve["Y_p"] = Y0
        self.x_axis_curve["X1_p"] = X0 + (X1_x - X0)/mt.cos(angle)
        self.x_axis_curve["X2_p"] = X0 + (X2_x - X0)/mt.cos(angle)

        self.y_axis_curve["X_p"] = X0
        self.y_axis_curve["Y1_p"] = Y0 + (Y1_y - Y0)/mt.cos(angle)
        self.y_axis_curve["Y2_p"] = Y0 + (Y2_y - Y0)/mt.cos(angle)
        self.update_lin_log()

    def update_lin_log(self):
        """
        Method to update the axis to a log or a linear and
        make the relation between graph and pixel space.
        """
        if self.app_state >= 3:
            if self.img_op.x_lin.isChecked():
                X1 = self.coord[0]
                X2 = self.coord[1]
            else:
                X1 = mt.log10(self.coord[0])
                X2 = mt.log10(self.coord[1])

            X1_p = self.x_axis_curve["X1_p"]
            X2_p = self.x_axis_curve["X2_p"]
            self.Xpr = (X2 - X1)/(X2_p - X1_p)

            if self.img_op.y_lin.isChecked():
                Y1 = self.coord[2]
                Y2 = self.coord[3]
            else:
                Y1 = mt.log10(self.coord[2])
                Y2 = mt.log10(self.coord[3])

            Y1_p = self.y_axis_curve["Y1_p"]
            Y2_p = self.y_axis_curve["Y2_p"]
            self.Ypr = (Y2 - Y1)/(Y2_p - Y1_p)
            self.set_formula()

    def update_image(self) -> None:
        """ Method to update the image with the contour chosen in the combobox """
        if self.app_state == 3:
            img = cv2.cvtColor(cv2.imread(ROTA_IMG), cv2.COLOR_BGR2GRAY)
            tr1, tr2 = [self.img_op.slider1.value(), self.img_op.slider2.value()]

            mode = self.img_op.combo.currentText()

            if mode == "Canny":
                img = cv2.Canny(img, tr1, tr2)
            elif mode == "Global Tresholding":
                img = cv2.medianBlur(img, 5)
                ret, img = cv2.threshold(img, tr1, tr2, cv2.THRESH_BINARY)
            elif mode == "Adaptive Mean Tresholding":
                img = cv2.medianBlur(img, 5)
                img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
            elif mode == "Adaptive Gausian Tresholding":
                img = cv2.medianBlur(img, 5)
                img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            elif mode == "Otsu's Tresholding":
                ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            elif mode == "Otsu's Tresholding + Gausian Blur":
                img = cv2.GaussianBlur(img, (5, 5), 0)
                ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            else:
                pass

            cont, h = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            a, b = img.shape
            img = np.zeros((a, b, 3))

            for c in cont:
                col = (randrange(255), randrange(255), randrange(255))
                cv2.drawContours(img, c, -1, col)

            cv2.imwrite(CONT_IMG, img)
            self.img.source = CONT_IMG

    def pixel_to_graph(self, pt: tuple) -> np.ndarray:
        """ Method to convert a pixel coordinate to a graph coordinate """
        x, y = pt
        if self.img_op.x_lin.isChecked():
            a = self.Xpr*(x - self.x_axis_curve["X1_p"]) + self.coord[0]
        else:
            a = mt.pow(10, self.Xpr*(x - self.x_axis_curve["X1_p"]) + mt.log10(self.coord[0]))
        if self.img_op.y_lin.isChecked():
            b = self.Ypr*(y - self.y_axis_curve["Y1_p"]) + self.coord[2]
        else:
            b = mt.pow(10, self.Ypr*(y - self.y_axis_curve["Y1_p"]) + mt.log10(self.coord[2]))
        return np.array([a, b])

    def graph_to_pixel(self, pt: tuple) -> np.ndarray:
        """ Method to convert a graph coordinate to a pixel coordinate """
        a, b = pt
        if self.img_op.x_lin:
            x = (a - self.coord[0])/self.Xpr + self.x_axis_curve["X1_p"]
        else:
            x = (mt.log10(a) - mt.log10(self.coord[0]))/self.Xpr + self.x_axis_curve["X1_p"]
        if self.img_op.y_lin:
            y = (b - self.coord[2])/self.Ypr + self.y_axis_curve["Y1_p"]
        else:
            y = (mt.log10(b) - mt.log10(self.coord[2]))/self.Ypr + self.y_axis_curve["Y1_p"]
        return np.array([x, y])

    def set_formula(self, do: bool = True) -> None:
        """ Method to update the formula displayed in the instruction box """
        if do and self.app_state >= 5:
            if self.img_op.x_lin.isChecked():
                x = np.array(self.pts_final_r)[:, 0]
                x_log = False
            else:
                x = np.log10(self.pts_final_r)[:, 0]
                x_log = True
            if self.img_op.y_lin.isChecked():
                y = np.array(self.pts_final_r)[:, 1]
                y_log = False
            else:
                y = np.log10(self.pts_final_r)[:, 1]
                y_log = True

            if self.img_op.y_from_x.isChecked():
                var = "x"
                a = x
                b = y
                a_log = x_log
                b_log = y_log
            else:
                var = "y"
                a = y
                b = x
                a_log = y_log
                b_log = x_log

            self.var = var
            self.islog = [a_log, b_log]
            self.order = order = self.img_op.spinbox.value()
            self.coef = coef = np.polyfit(a, b, order)
            b = np.poly1d(coef)
            eval_a = np.linspace(min(a), max(a), 100)
            eval_b = b(eval_a)

            if a_log:
                eval_a = np.power(10, eval_a)
            if b_log:
                eval_b = np.power(10, eval_b)

            if var == "x":
                ex = eval_a
                ey = eval_b
            else:
                ex = eval_b
                ey = eval_a

            eval_pts = []
            for (x, y) in zip(ex, ey):
                eval_pts.append(np.array([x, y]))

            self.pts_eval_r = eval_pts

            formula = ""
            if a_log:
                var = "(log<sub>10</sub>{0})".format(var)
            if b_log:
                formula += "10^("
            for (i, c) in enumerate(coef):
                if order - i > 1:
                    formula += "{0:+.2e} {1}<sup>{2}</sup> ".format(c, var, order - i)
                elif order - i == 1:
                    formula += "{0:+.2e} {1} ".format(c, var)
                else:
                    formula += "{0:+.2e}".format(c)

            if b_log:
                formula += ")"
            text = "The formula for this curve is :\n\n" \
                   "{0}\n\nFor more precision, use the copy function below.".format(formula)
            self.instruct.textbox.setMarkdown(text)

            if self.app_state == 6:
                self.plot_points()

    def plot_points(self) -> None:
        """ Method to generate the plot image and display it """
        fig = Figure(figsize=(11.0, 6.25), dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.gca()

        x_true, y_true = [np.array(self.pts_final_r)[:, 0], np.array(self.pts_final_r)[:, 1]]
        x_eval, y_eval = [np.array(self.pts_eval_r)[:, 0], np.array(self.pts_eval_r)[:, 1]]
        ax.plot(x_true, y_true, 'or', label="Extracted")
        ax.plot(x_eval, y_eval, '-b', label="Evaluated")
        ax.legend()
        ax.grid()

        if self.img_op.x_lin.isChecked():
            ax.set_xscale('linear')
        else:
            ax.set_xscale('log')
        if self.img_op.y_lin.isChecked():
            ax.set_yscale('linear')
        else:
            ax.set_yscale('log')

        canvas.draw()  # draw the canvas, cache the renderer

        img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(MAX_IMG_H, MAX_IMG_W, 3)
        cv2.imwrite(PLOT_IMG, img)
        self.img.source = PLOT_IMG

    def copy_text(self) -> None:
        """ Method to copy certain data """
        mode = self.instruct.combo.currentText()
        var = self.var
        a_log, b_log = self.islog

        if mode == "Copy Formula - Matlab":
            formula = ""
            if a_log:
                var = "(log10({0}))".format(var)
            if b_log:
                formula += "10.^("
            for (i, c) in enumerate(self.coef):
                if self.order - i > 1:
                    formula += "+ {0}*{1}.^{2} ".format(c, var, self.order - i)
                elif self.order - i == 1:
                    formula += "+ {0}*{1} ".format(c, var)
                else:
                    formula += "+ {0}".format(c)
            if b_log:
                formula += ")"
            text = formula

        elif mode == "Copy Formula - Python":
            formula = ""
            if a_log:
                var = "(np.log10({0}))".format(var)
            if b_log:
                formula += "np.power(10, "
            for (i, c) in enumerate(self.coef):
                if self.order - i > 1:
                    formula += "+ {0}*{1}**({2}) ".format(c, var, self.order - i)
                elif self.order - i == 1:
                    formula += "+ {0}*{1} ".format(c, var)
                else:
                    formula += "+ {0}".format(c)
            if b_log:
                formula += ")"
            text = formula

        elif mode == "Copy Formula - Markdown":
            formula = ""
            if a_log:
                var = "(log<sub>10</sub>{0})".format(var)
            if b_log:
                formula += "10^("
            for (i, c) in enumerate(self.coef):
                if self.order - i > 1:
                    formula += "{0:+.2e} {1}<sup>{2}</sup> ".format(c, var, self.order - i)
                elif self.order - i == 1:
                    formula += "{0:+.2e} {1} ".format(c, var)
                else:
                    formula += "{0:+.2e}".format(c)
            if b_log:
                formula += ")"
            text = formula

        elif mode == "Copy Points - Matlab":
            text = "x = ["
            x_r = []
            y_r = []

            for d in self.pts_final_r:
                x_r.append(d[0])
                y_r.append(d[1])

            for (i, x) in enumerate(x_r):
                if i == 0:
                    text += "{0}".format(x)
                else:
                    text += " {0}".format(x)

            text += "];\ny = ["
            for (i, y) in enumerate(y_r):
                if i == 0:
                    text += "{0}".format(y)
                else:
                    text += " {0}".format(y)

            text += "];"

        elif mode == "Copy Points - Python":
            text = "x = ["
            x_r = []
            y_r = []

            for d in self.pts_final_r:
                x_r.append(d[0])
                y_r.append(d[1])

            for (i, x) in enumerate(x_r):
                if i == 0:
                    text += "{0}".format(x)
                else:
                    text += ", {0}".format(x)

            text += "];\ny = ["
            for (i, y) in enumerate(y_r):
                if i == 0:
                    text += "{0}".format(y)
                else:
                    text += ", {0}".format(y)

            text += "];"

        elif mode == "Copy Points - NumPy":
            text = "pts = np.array([["
            x_r = []
            y_r = []

            for d in self.pts_final_r:
                x_r.append(d[0])
                y_r.append(d[1])

            for (i, x) in enumerate(x_r):
                if i == 0:
                    text += "{0}".format(x)
                else:
                    text += ", {0}".format(x)

            text += "], ["
            for (i, y) in enumerate(y_r):
                if i == 0:
                    text += "{0}".format(y)
                else:
                    text += ", {0}".format(y)

            text += "]])"

        elif mode == "Copy Points - CSV":
            text = "x, y\n"

            for d in self.pts_final_r:
                text += "{0}, {1}\n".format(d[0], d[1])

        elif mode == "Copy Coeff. - Matlab":
            text = "["
            for (i, coef) in enumerate(self.coef):
                if i == 0:
                    text += "{0}".format(coef)
                else:
                    text += " {0}".format(coef)
            text += "]"

        elif mode == "Copy Coeff. - Python":
            text = "["
            for (i, coef) in enumerate(self.coef):
                if i == 0:
                    text += "{0}".format(coef)
                else:
                    text += ", {0}".format(coef)
            text += "]"

        elif mode == "Copy Coeff. - NumPy":
            text = "np.array(["
            for (i, coef) in enumerate(self.coef):
                if i == 0:
                    text += "{0}".format(coef)
                else:
                    text += ", {0}".format(coef)
            text += "])"

        elif mode == "Copy Poly1D - NumPy":
            text = "p = np.poly1d(["
            for (i, coef) in enumerate(self.coef):
                if i == 0:
                    text += "{0}".format(coef)
                else:
                    text += ", {0}".format(coef)
            text += "])"

        else:
            return

        QApplication.clipboard().setText(text)

    @property
    def app_state(self) -> int:
        """ Method to get the current app state """
        return self._app_state

    @app_state.setter
    def app_state(self, state: int) -> None:
        """ Method where the sequence of the app is handled """
        self._app_state = state

        if state == 0:
            """Starting state"""
            self.instruct.textbox.setMarkdown("Choose a picture of a graph and press `Start`")
            self.but_start.setText("Start")
            self.but_next.setText("Next")
            self.coord_prompt.initValues()
            self.instruct.setEnabled(False)
            self.img_op.setEnabled(True)
            self.img_op.is_brush = True
        elif state == 1:
            """Pressed Start"""
            self.instruct.textbox.setMarkdown("1) Click on 2 points for each axis in this order:\n"
                                              "    **X1 -> X2 -> Y1 -> Y2**\n"
                                              "2) Enter the coordinates in the box below.")
            self.but_start.setText("Restart")
            self.but_next.setText("Next")
            self.coord_prompt.initValues()
            self.instruct.setEnabled(False)
            self.img_op.setEnabled(True)
            self.img_op.is_brush = True
            self.pts_final_p = []
            self.pts_final_r = []
            self.pts_eval_r = []
            self.img.source = ORIG_IMG
        elif state == 2:
            """Coordinate all selected"""
        elif state == 3:
            """Chose the coord and rotated"""
            self.resize_and_rotate()
            self.update_image()
            self.instruct.textbox.setMarkdown("Adjust the thresholding so that the curve you want to extract "
                                              "is clearly visible and free of close obstacles.\n\n"
                                              "When done, press `Next`")
        elif state == 4:
            """Chose the displaying"""
            self.img_op.setEnabled(False)
            self.img.holdEnabled = True
            img = cv2.cvtColor(cv2.imread(CONT_IMG), cv2.COLOR_BGR2GRAY)
            img = np.greater(img, np.zeros(img.shape))*255  # Create the contour mask
            self.mask = np.ones(img.shape)
            cv2.imwrite(CTMK_IMG, img)
            self.instruct.textbox.setMarkdown("Press and hold over the curve you want to extract. All the "
                                              '"painted" multi-colored points will be extracted.'
                                              "When you selected all the curve, press `Next` to extract "
                                              "the data points.")
        elif state == 5:
            """Selected the edges to keep"""
            img = cv2.cvtColor(cv2.imread(CTMK_IMG), cv2.COLOR_BGR2GRAY)
            img = np.equal(img, self.mask)
            pts_y, pts_x = np.where(img)

            img = cv2.imread(ROTA_IMG)
            for (x, y) in zip(pts_x, pts_y):
                a, b = self.pixel_to_graph((x, y))
                self.pts_final_p.append((x, y))
                self.pts_final_r.append(np.array([a, b]))
                cv2.circle(img, (x, y), 2, (0, 0, 255), -1)

            rad = int(self.img.original_image_size[0] / 100)
            cv2.circle(img, (int(self.x_axis_curve["X1_p"]), int(self.x_axis_curve["Y_p"])),
                       rad, self.pts_colors[0], -1)
            cv2.circle(img, (int(self.x_axis_curve["X2_p"]), int(self.x_axis_curve["Y_p"])),
                       rad, self.pts_colors[1], -1)
            cv2.circle(img, (int(self.y_axis_curve["X_p"]), int(self.y_axis_curve["Y1_p"])),
                       rad, self.pts_colors[2], -1)
            cv2.circle(img, (int(self.y_axis_curve["X_p"]), int(self.y_axis_curve["Y2_p"])),
                       rad, self.pts_colors[3], -1)

            cv2.imwrite(SELE_IMG, img)
            self.img.source = SELE_IMG
            self.instruct.setEnabled(True)
            self.img_op.is_brush = False
            self.set_formula()
            self.but_next.setText("Plot")
        elif state == 6:
            """Ready to plot"""
            self.but_next.setText("Image")
            self.plot_points()


if __name__ == "__main__":
    seed(123456)
    app = QApplication([])
    window = CurveFinder()
    window.show()
    sys.exit(app.exec_())
