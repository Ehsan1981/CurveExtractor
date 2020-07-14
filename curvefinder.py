# This Python file uses the following encoding: utf-8
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QHBoxLayout, QVBoxLayout, QPushButton, QFileDialog,\
    QCheckBox, QLineEdit, QMessageBox, QTextEdit, QSlider, QComboBox, QRadioButton, QSpinBox
from PyQt5.QtGui import QPixmap, QMouseEvent, QFont, QIcon
from PyQt5.QtCore import Qt, pyqtSignal

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import random as rand
import numpy as np
import math as mt
import shutil
import cv2
import sys
import os


VER = "2.0"
AUTHOR = "Bruno-P. Busque"

MAX_IMG_W = 1100
MAX_IMG_H = 625

ICON_PATH = "icon.png"
PH_IMAGE_PATH = "placeholder.png"
TEMP_PATH = "temp/"
ORIG_IMG = TEMP_PATH + "original_img.png"
COOR_IMG = TEMP_PATH + "coordinate_img.png"
ROTA_IMG = TEMP_PATH + "rotated_img.png"
CONT_IMG = TEMP_PATH + "contoured_img.png"
CTMK_IMG = TEMP_PATH + "masked_contoured_img.png"
SELE_IMG = TEMP_PATH + "selected_img.png"
PLOT_IMG = TEMP_PATH + "plotted_img.png"


class QImage(QLabel):
    signal = pyqtSignal(int, int)
    holdEnabled = False
    holding = False

    def __init__(self, image_path: str):
        super().__init__()
        self.setStyleSheet("border: 3px solid gray;")
        self.img_path = image_path
        self.source = image_path

    @property
    def source(self):
        return self._source

    @source.setter
    def source(self, src):
        self.img_path = src
        new_img = QPixmap(src)
        self.original_image_size = (new_img.height(), new_img.width())
        self._source = new_img.scaled(MAX_IMG_W, MAX_IMG_H, Qt.KeepAspectRatio)
        self.new_image_size = (self._source.height(), self._source.width())
        self.setPixmap(self._source)

    def mousePressEvent(self, ev: QMouseEvent):
        self.holding = True
        x, y = [self.original_image_size[i]*(ev.x(), ev.y())[i]/self.new_image_size[i] for i in (0, 1)]
        self.signal.emit(x, y)

    def mouseMoveEvent(self, ev: QMouseEvent):
        if self.holding and self.holdEnabled:
            x, y = [self.original_image_size[i] * (ev.x(), ev.y())[i] / self.new_image_size[i] for i in (0, 1)]
            self.signal.emit(x, y)

    def mouseReleaseEvent(self, ev: QMouseEvent):
        self.holding = False


class QCoord(QVBoxLayout):

    pts_labels = ["X1", "X2", "Y1", "Y2"]
    pts = [(-1, -1)]*4

    def __init__(self):
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

    def initValues(self):
        self.x1_done = False
        self.x2_done = False
        self.y1_done = False
        self.y2_done = False

    @property
    def x1_done(self):
        return self._x1_done

    @x1_done.setter
    def x1_done(self, status):
        self._x1_done = status
        self.x1_coord.check.setChecked(status)
        if not status:
            self.pts[0] = (-1, -1)

    @property
    def x2_done(self):
        return self._x2_done

    @x2_done.setter
    def x2_done(self, status):
        self._x2_done = status
        self.x2_coord.check.setChecked(status)
        if not status:
            self.pts[1] = (-1, -1)

    @property
    def y1_done(self):
        return self._y1_done

    @y1_done.setter
    def y1_done(self, status):
        self._y1_done = status
        self.y1_coord.check.setChecked(status)
        if not status:
            self.pts[2] = (-1, -1)

    @property
    def y2_done(self):
        return self._y2_done

    @y2_done.setter
    def y2_done(self, status):
        self._y2_done = status
        self.y2_coord.check.setChecked(status)
        #self.but_next.setEnabled(status)
        if not status:
            self.pts[3] = (-1, -1)
        #else:
        #    self.instruct.setMarkdown("Enter the graph coordinates for each points.\n"
        #                              "\n"
        #                              "Press `Next` if your satisfied with your points.\n"
        #                              "Press `Restart` if not.")

    class QCoordBox(QHBoxLayout):

        def __init__(self, coord_label: str):
            super().__init__()

            # Set the widgets
            self.coord_label = coord_label
            self.label = QLabel(text="{0} :".format(coord_label))
            self.line = QLineEdit()
            self.line.setPlaceholderText("Enter coord. for {0}...".format(coord_label))
            self.check = QCheckBox(text="{0} placed".format(coord_label))
            self.check.setEnabled(False)

            # Create the layout
            self.addWidget(self.label)
            self.addWidget(self.line)
            self.addWidget(self.check)


class QInstructBox(QVBoxLayout):

    options = ["Copy formula (Markdown)", "Copy retrieved points", "Copy Coef. for Numpy"]

    def __init__(self):
        super().__init__()

        self.combo = QComboBox()
        self.combo.addItems(self.options)
        self.but_copy = QPushButton(text="Copy")

        hbox = QHBoxLayout()
        hbox.addWidget(self.combo, stretch=1)
        hbox.addWidget(self.but_copy)

        self.textbox = QTextEdit()
        self.textbox.setMarkdown("Select a graph to start.")
        self.textbox.setEnabled(False)
        self.textbox.setFont(QFont("Calibri", 12, QFont.Bold))

        vbox = QVBoxLayout()
        vbox.addWidget(self.textbox)
        vbox.addLayout(hbox)

        self.addLayout(vbox)

    def setEnabled(self, a0: bool):
        self.combo.setEnabled(a0)
        self.but_copy.setEnabled(a0)


class QImageOptions(QVBoxLayout):

    options = ["Canny", "Global Tresholding", "Adaptive Mean Tresholding", "Adaptive Gausian Tresholding",
               "Otsu's Tresholding", "Otsu's Tresholding + Gausian Blur"]
    treshs = [(True, True), (True, True), (False, False), (False, False),
              (False, False), (False, False), (False, False)]
    tresh_ext = [(-1000, 1000, -1000, 1000), (0, 255, 0, 255), (0, 1, 0, 1), (0, 1, 0, 1),
                 (0, 1, 0, 1), (0, 1, 0, 1), (0, 1, 0, 1)]

    def __init__(self):
        super().__init__()

        # Set the radios
        self.label0 = QLabel(text="Wanted formula :")
        self.y_from_x = QRadioButton(text="y = f(x)")
        self.y_from_x.setChecked(True)
        self.x_from_y = QRadioButton(text="x = f(y)")

        hb0 = QHBoxLayout()
        hb0.addWidget(self.label0)
        hb0.addStretch(1)
        hb0.addWidget(self.y_from_x)
        hb0.addWidget(self.x_from_y)

        # Set the combobox
        self.combo = QComboBox()
        self.combo.addItems(self.options)
        self.combo.currentTextChanged.connect(self.combo_change)

        # Set the first labeled slider
        self.label1 = QLabel(text="Tresh. 1")
        self.slider1 = QSlider(Qt.Horizontal)
        self.slider1.setMinimum(-1000)
        self.slider1.setMaximum(1000)
        self.slider1.setTickInterval(100)
        self.slider1.setTickPosition(QSlider.TicksBelow)

        hb1 = QHBoxLayout()
        hb1.addWidget(self.label1)
        hb1.addWidget(self.slider1)

        # Set the second labeled slider
        self.label2 = QLabel(text="Tresh. 2")
        self.slider2 = QSlider(Qt.Horizontal)
        self.slider2.setMinimum(-1000)
        self.slider2.setMaximum(1000)
        self.slider2.setTickInterval(100)
        self.slider2.setTickPosition(QSlider.TicksBelow)

        hb2 = QHBoxLayout()
        hb2.addWidget(self.label2)
        hb2.addWidget(self.slider2)

        # Set the brush/order spinbox
        self.label3 = QLabel()
        self.spinbox = QSpinBox()
        self.spinbox.setMinimum(0)
        self.is_brush = True
        self.spinbox.valueChanged.connect(self.update_brush_label)

        hb3 = QHBoxLayout()
        hb3.addWidget(self.label3)
        hb3.addWidget(self.spinbox)

        # Set the final layout
        self.addLayout(hb0)
        self.addWidget(self.combo)
        self.addLayout(hb1)
        self.addLayout(hb2)
        self.addLayout(hb3)

    def combo_change(self, text):
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

    def setEnabled(self, a0: bool):
        self.combo.setEnabled(a0)
        self.slider1.setEnabled(a0)
        self.slider2.setEnabled(a0)

    def update_brush_label(self):
        if not self.is_brush:
            self.label3.setText("Fit order :".format(self.spinbox.value()))

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

    img_src = PH_IMAGE_PATH
    coord = np.zeros(4, dtype=float)
    pts_colors = [(204, 0, 0), (0, 153, 0), (0, 0, 153), (204, 204, 0)]
    pts_labels = ["X1", "X2", "Y1", "Y2"]
    pts_final_p = []
    pts_final_r = []
    pts_eval_r = []
    curve1 = {}
    curve2 = {}
    Xpr = 0
    Ypr = 0
    coef = []
    order = 5
    var = "x"
    xlin = True  # Implement log
    ylin = True  # Implement log
    mask = None

    def __init__(self):
        QWidget.__init__(self)
        self.setWindowTitle("Curve Finder Ver. {0}".format(VER))
        self.setWindowIcon(QIcon(ICON_PATH))
        self.setFixedWidth(1500)
        self.setFixedHeight(750)

        # Create a temporary folder
        if not os.path.exists(TEMP_PATH):
            os.mkdir(TEMP_PATH)

        # Create widgets
        title_label = QLabel(text="Curve Finder Ver. {0} by {1}".format(VER, AUTHOR))
        title_label.setFont(QFont("Calibri", 20, QFont.Bold))
        self.img = QImage(PH_IMAGE_PATH)
        self.instruct = QInstructBox()
        self.img_op = QImageOptions()
        self.coord_prompt = QCoord()
        self.but_browse = QPushButton(text="Select an image")
        self.but_start = QPushButton(text="Start")
        self.but_next = QPushButton(text="Next")

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

        hbox = QHBoxLayout()
        hbox.addWidget(self.img, alignment=Qt.AlignCenter, stretch=4)
        hbox.addLayout(options, stretch=1)

        vbox = QVBoxLayout()
        vbox.addWidget(title_label)
        vbox.addLayout(hbox)
        self.setLayout(vbox)

        # Set application state
        self.app_state = 0

        self.show()

    def __del__(self):
        # Remove the temporary folder
        shutil.rmtree(TEMP_PATH)

    def browse_for_image(self):
        self.img_src = str(QFileDialog().getOpenFileName(filter="Images (*.png *.bmp *.jpg)")[0])
        if self.img_src != "":
            self.img.source = self.img_src
            self.app_state = 0  # Return to initial state

    def start(self):
        self.instruct.textbox.setMarkdown("Click on 2 points for each axis in this order:\n"
                                          "```\n - X1\n - X2\n - Y1\n - Y2\n```")
        cv2.imwrite(ORIG_IMG, cv2.imread(self.img_src))
        self.app_state = 1

    def next(self):
        if self.app_state == 2 and self.verify_coord():
            self.app_state = 3

        elif self.app_state == 3:
            self.app_state = 4

        elif self.app_state == 4:
            self.app_state = 5

        elif self.app_state == 5:
            self.app_state = 6

    def verify_coord(self):
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

    def add_position(self, x: int, y: int):
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

    def draw_points(self, x1: tuple = None, x2: tuple = None, y1: tuple = None, y2: tuple = None):
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
                cv2.circle(img, pt, 5, self.pts_colors[i], -1)
                cv2.putText(img, self.pts_labels[i], pt, cv2.FONT_HERSHEY_SIMPLEX, 1, self.pts_colors[i], 2)

        cv2.imwrite(COOR_IMG, img)
        self.img.source = COOR_IMG

    def draw_mask(self, x: int, y: int):
        alpha = 0.3
        radius = self.img_op.spinbox.value()
        img = cv2.imread(CONT_IMG)
        new_img = img.copy()
        cv2.circle(new_img, (x, y), radius, (0, 0, 255), -1)
        cv2.addWeighted(new_img, alpha, img, 1 - alpha, 0, img)
        cv2.circle(self.mask, (x, y), radius, 255, -1)
        cv2.imwrite(CONT_IMG, img)
        self.img.source = CONT_IMG

    def resize_and_rotate(self):
        if self.xlin:
            X1 = self.curve1["X1"] = self.coord_prompt.pts[0][0]
            X2 = self.curve1["X2"] = self.coord_prompt.pts[1][0]
            XL = self.curve2["XL"] = self.coord_prompt.pts[2][0]
            dX = self.curve2["dX"] = self.coord_prompt.pts[3][0] - XL
        else:
            X1 = self.curve1["X1"] = mt.log10(self.coord_prompt.pts[0][0])
            X2 = self.curve1["X2"] = mt.log10(self.coord_prompt.pts[1][0])
            XL = self.curve2["XL"] = mt.log10(self.coord_prompt.pts[2][0])
            dX = self.curve2["dX"] = mt.log10(self.coord_prompt.pts[3][0]) - XL

        if self.ylin:
            Y1 = self.curve2["Y1"] = self.coord_prompt.pts[2][1]
            Y2 = self.curve2["Y2"] = self.coord_prompt.pts[3][1]
            YL = self.curve1["YL"] = self.coord_prompt.pts[0][1]
            dY = self.curve1["dY"] = self.coord_prompt.pts[1][1] - YL
        else:
            Y1 = self.curve2["Y1"] = mt.log10(self.coord_prompt.pts[2][1])
            Y2 = self.curve2["Y2"] = mt.log10(self.coord_prompt.pts[3][1])
            YL = self.curve1["YL"] = mt.log10(self.coord_prompt.pts[0][1])
            dY = self.curve1["dY"] = mt.log10(self.coord_prompt.pts[1][1]) - YL

        A1 = self.curve1["A1"] = dY / (X2 - X1)
        A2 = self.curve2["A2"] = (Y2 - Y1) / dX

        X0 = int((A1 * X1 - A2 * XL + Y1 - YL) / (A1 - A2))
        Y0 = int(A1 * (X0 - X1) + YL)
        origin = (X0, Y0)
        angle = theta = (mt.atan(dY / (X2 - X1)) + mt.atan(-dX / (Y2 - Y1))) / 2

        pts_prime = np.zeros(4, dtype=tuple)
        pts_prime[0] = (int(X0 + (X1 - X0) / mt.cos(theta)), Y0)
        pts_prime[1] = (int(X0 + (X2 - X0) / mt.cos(theta)), Y0)
        pts_prime[2] = (X0, int(Y0 + (Y1 - Y0) / mt.cos(theta)))
        pts_prime[3] = (X0, int(Y0 + (Y2 - Y0) / mt.cos(theta)))

        self.Xpr = (self.coord[1] - self.coord[0]) / (X2 - X1)
        self.Ypr = (self.coord[3] - self.coord[2]) / (Y2 - Y1)

        M = cv2.getRotationMatrix2D(origin, 180 * angle / mt.pi, 1)
        img = cv2.imread(self.img_src)
        img = cv2.warpAffine(img, M, img.shape[1::-1], flags=cv2.INTER_LINEAR)

        cv2.imwrite(ROTA_IMG, img)
        self.img.source = ROTA_IMG

    def update_image(self):
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
                col = (rand.randrange(255), rand.randrange(255), rand.randrange(255))
                cv2.drawContours(img, c, -1, col)

            cv2.imwrite(CONT_IMG, img)
            self.img.source = CONT_IMG

    def pixel_to_graph(self, pt):
        x, y = pt
        a = self.Xpr * (x - self.curve1["X1"]) + self.coord[0]
        b = self.Ypr * (y - self.curve2["Y1"]) + self.coord[2]
        return np.array([a, b])

    def graph_to_pixel(self, pt):
        a, b = pt
        x = (a - self.coord[0])/self.Xpr + self.curve1["X1"]
        y = (b - self.coord[2])/self.Ypr + self.curve2["Y1"]
        return np.array([x, y])

    def set_formula(self, do: bool = True):
        if do and self.app_state >= 5:
            x, y = [np.array(self.pts_final_r)[:, 0], np.array(self.pts_final_r)[:, 1]]
            if self.img_op.y_from_x.isChecked():
                var = "x"
                a = x
                b = y
            else:
                var = "y"
                a = y
                b = x

            self.var = var
            self.order = order = self.img_op.spinbox.value()
            self.coef = coef = np.polyfit(a, b, order)
            b = np.poly1d(coef)
            eval_a = np.linspace(min(a), max(a), 100)
            eval_b = b(eval_a)

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
            for (i, c) in enumerate(coef):
                if order - i > 1:
                    formula += "{0:+.2e} {1}<sup>{2}</sup> ".format(c, var, order - i)
                elif order - i == 1:
                    formula += "{0:+.2e} {1} ".format(c, var)
                else:
                    formula += "{0:+.2e}".format(c)

            text = "The formula for this curve is :\n\n{0}".format(formula)
            self.instruct.textbox.setMarkdown(text)

            if self.app_state == 6:
                self.plot_points()

    def plot_points(self):
        fig = Figure(figsize=(11.0, 6.25), dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.gca()

        x_true, y_true = [np.array(self.pts_final_r)[:, 0], np.array(self.pts_final_r)[:, 1]]
        x_eval, y_eval = [np.array(self.pts_eval_r)[:, 0], np.array(self.pts_eval_r)[:, 1]]
        ax.plot(x_true, y_true, 'or')
        ax.plot(x_eval, y_eval, '-b')
        ax.legend(["True", "Evaluated"])
        ax.grid()

        canvas.draw()  # draw the canvas, cache the renderer

        img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(MAX_IMG_H, MAX_IMG_W, 3)
        cv2.imwrite(PLOT_IMG, img)
        self.img.source = PLOT_IMG

    def copy_text(self):
        mode = self.instruct.combo.currentText()

        if mode == "Copy formula (Markdown)":
            formula = ""
            for (i, c) in enumerate(self.coef):
                if self.order - i > 1:
                    formula += "{0:+.2e} {1}<sup>{2}</sup> ".format(c, self.var, self.order - i)
                elif self.order - i == 1:
                    formula += "{0:+.2e} {1} ".format(c, self.var)
                else:
                    formula += "{0:+.2e}".format(c)
            text = formula
        elif mode == "Copy retrieved points":
            text = str(self.pts_final_r)
        elif mode == "Copy Coef. for Numpy":
            text = "["
            for (i, coef) in enumerate(self.coef):
                if i == 0:
                    text += "{0}".format(coef)
                else:
                    text += ", {0}".format(coef)
            text += "]"
        else:
            return

        QApplication.clipboard().setText(text)

    @property
    def app_state(self):
        return self._app_state

    @app_state.setter
    def app_state(self, state):
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
            self.instruct.textbox.setMarkdown("Adjust the thresholding so that the curve you want to extract"
                                      " is clearly visible.\n\n"
                                      "When done, press `Next`")
        elif state == 4:
            """Chose the displaying"""
            self.img_op.setEnabled(False)
            self.img.holdEnabled = True
            img = cv2.cvtColor(cv2.imread(CONT_IMG), cv2.COLOR_BGR2GRAY)
            img = np.greater(img, np.zeros(img.shape))*255  # Create the contour mask
            self.mask = np.ones(img.shape)
            cv2.imwrite(CTMK_IMG, img)
            self.instruct.textbox.setMarkdown("Press and hold over the curve you want to extract. "
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

            cv2.imwrite(SELE_IMG, img)
            self.img.source = SELE_IMG
            self.instruct.setEnabled(True)
            self.img_op.is_brush = False
            self.set_formula()
            self.but_next.setText("Plot")
        elif state == 6:
            """Ready to plot"""
            self.plot_points()


if __name__ == "__main__":
    rand.seed(123456)
    app = QApplication([])
    window = CurveFinder()
    window.show()
    sys.exit(app.exec_())
