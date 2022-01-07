from typing import Union, List, Tuple
from numpy import ndarray, array
from enum import IntEnum
from .constants import *
import math as mt
import cv2


class CurveFinder:

    def __init__(self) -> None:
        self.graph = self.Graph()
        self.X1: float = 0
        self.X2: float = 0
        self.Y1: float = 0
        self.Y2: float = 0
        self.Xpr: float = 0
        self.Ypr: float = 0

    def get_rotation_matrix(self, pts: List[Tuple[int, int]]) -> ndarray:
        self.graph.x_axis.pts = ((pts[0][0], pts[0][1]), (pts[1][0], pts[1][1]))
        self.graph.y_axis.pts = ((pts[2][0], pts[2][1]), (pts[3][0], pts[3][1]))
        self.graph.update()

        return cv2.getRotationMatrix2D(self.graph.origin, 180*self.graph.angle/mt.pi, 1)

    def update_lin_log(self, x_is_lin: bool, y_is_lin: bool):
        """
        Method to update the axis to a log or a linear and
        make the relation between graph and pixel space.
        """
        if x_is_lin:
            x1 = self.X1
            x2 = self.X2
        else:
            x1 = mt.log10(self.X1)
            x2 = mt.log10(self.X2)

        dx_p = self.graph.x_axis.X1_p - self.graph.x_axis.X2_p
        self.Xpr = (x2 - x1)/dx_p

        if y_is_lin:
            y1 = self.Y1
            y2 = self.Y2
        else:
            y1 = mt.log10(self.Y1)
            y2 = mt.log10(self.Y2)

        dy_p = self.graph.y_axis.Y1_p - self.graph.y_axis.Y2_p
        self.Ypr = (y2 - y1)/dy_p

    def get_points(self) -> Tuple[Tuple[float, float], ...]:
        y_p = self.graph.x_axis.Y_p
        x1_p = self.graph.x_axis.X1_p
        x2_p = self.graph.x_axis.X2_p
        x_p = self.graph.y_axis.X_p
        y1_p = self.graph.y_axis.Y1_p
        y2_p = self.graph.y_axis.Y2_p
        return (x1_p, y_p), (x2_p, y_p), (x_p, y1_p), (x_p, y2_p)

    def pixel_to_graph(self, pt: tuple, x_is_lin: bool, y_is_lin: bool) -> ndarray:
        """ Method to convert a pixel coordinate to a graph coordinate """
        x, y = pt

        if x_is_lin:
            a = self.Xpr*(x - self.graph.x_axis.X1_p) + self.X1
        else:
            a = mt.pow(10, self.Xpr*(x - self.graph.x_axis.X1_p) + mt.log10(self.X1))

        if y_is_lin:
            b = self.Ypr*(y - self.graph.y_axis.Y1_p) + self.Y1
        else:
            b = mt.pow(10, self.Ypr*(y - self.graph.y_axis.Y1_p) + mt.log10(self.Y1))

        return array([a, b])

    def graph_to_pixel(self, pt: tuple, x_is_lin: bool, y_is_lin: bool) -> ndarray:
        """ Method to convert a graph coordinate to a pixel coordinate """
        a, b = pt

        if x_is_lin:
            x = (a - self.X1)/self.Xpr + self.graph.x_axis.X1_p
        else:
            x = (mt.log10(a) - mt.log10(self.X1))/self.Xpr + self.graph.x_axis.X1_p

        if y_is_lin:
            y = (b - self.Y1)/self.Ypr + self.graph.y_axis.Y1_p
        else:
            y = (mt.log10(b) - mt.log10(self.Y1))/self.Ypr + self.graph.y_axis.Y1_p

        return array([x, y])

    class Graph:
        axis_corner: Tuple[float, float]

        def __init__(self) -> None:
            self.x_axis = self.Axis(self.Axis.AxisType.X)
            self.y_axis = self.Axis(self.Axis.AxisType.Y)
            self.angle: float = None
            self.origin: Tuple[float, float] = None

        def update(self) -> None:
            # Update the angle
            self.angle = (self.x_axis.angle + self.y_axis.angle)/2

            # Update the origin
            if self.y_axis.dx == 0:
                x0 = self.y_axis.pts[1][0]
            else:
                x0 = (self.y_axis.slope*self.y_axis.pts[0][0] - self.x_axis.slope*self.x_axis.pts[0][0]
                      + self.x_axis.pts[0][1] - self.y_axis.pts[0][1])/(self.y_axis.slope - self.x_axis.slope)
            y0 = self.x_axis.slope*(x0 - self.x_axis.pts[0][0]) + self.x_axis.pts[0][1]
            self.origin = (x0, y0)

            # Update axis attributes
            self.x_axis.Y_p = y0
            self.x_axis.X1_p = x0 + (self.x_axis.pts[0][0] - x0)/mt.cos(self.angle)
            self.x_axis.X2_p = x0 + (self.x_axis.pts[1][0] - x0)/mt.cos(self.angle)

            self.y_axis.X_p = x0
            self.y_axis.Y1_p = y0 + (self.y_axis.pts[0][1] - y0)/mt.cos(self.angle)
            self.y_axis.Y2_p = y0 + (self.y_axis.pts[1][1] - y0)/mt.cos(self.angle)

        class Axis:
            class AxisType(IntEnum):
                X = 0
                Y = 1

            def __init__(self, axis_type: AxisType):
                self.type = axis_type
                self.pts: Tuple[Tuple[int, int], Tuple[int, int]] = ((0, 0), (0, 1))
                self.slope: float
                self.angle: float
                self.Y_p: float = None
                self.X_p: float = None
                self.X1_p: float = None
                self.X2_p: float = None
                self.Y1_p: float = None
                self.Y2_p: float = None

            @property
            def pts(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
                return self._pts

            @pts.setter
            def pts(self, pt: Tuple[Tuple[int, int], Tuple[int, int]]) -> None:
                self._pts = pt
                self.dx = pt[1][0] - pt[0][0]
                self.dy = pt[1][1] - pt[0][1]
                self.slope = mt.inf if self.dx == 0 else self.dy/self.dx
                if self.type == self.AxisType.X:
                    self.angle = mt.atan(self.slope)
                elif self.type == self.AxisType.Y:
                    self.angle = mt.atan(-1/self.slope)


def get_copy_text(mode: int, var: str, islog: List[bool], coefs: list, order: int,
                  pts: List[ndarray]) -> Union[str, None]:
    a_log, b_log = islog
    equation = ""

    if mode == CopyOptions.EQUATION_MATLAB:
        if a_log:
            var = f"(log10({var}))"
        if b_log:
            equation += "10.^("
        for (i, c) in enumerate(coefs):
            if order - i > 1:
                equation += f"+ {c}*{var}.^{order - i} "
            elif order - i == 1:
                equation += f"+ {c}*{var} "
            else:
                equation += f"+ {c}"
        if b_log:
            equation += ")"

        return equation

    elif mode == CopyOptions.EQUATION_PYTHON:
        if a_log:
            var = f"(np.log10({var}))"
        if b_log:
            equation += "np.power(10, "
        for (i, c) in enumerate(coefs):
            if order - i > 1:
                equation += f"+ {c}*{var}**({order - i}) "
            elif order - i == 1:
                equation += f"+ {c}*{var} "
            else:
                equation += f"+ {c}"
        if b_log:
            equation += ")"

        return equation

    elif mode == CopyOptions.EQUATION_MARKDOWN:
        if a_log:
            var = f"(log<sub>10</sub>{var})"
        if b_log:
            equation += "10^("
        for (i, c) in enumerate(coefs):
            if order - i > 1:
                equation += f"{c:+.2e} {var}<sup>{order - i}</sup> "
            elif order - i == 1:
                equation += f"{c:+.2e} {var} "
            else:
                equation += f"{c:+.2e}"
        if b_log:
            equation += ")"

        return equation

    elif mode == CopyOptions.POINTS_MATLAB:
        text = "x = ["
        x_r = []
        y_r = []

        for d in pts:
            x_r.append(d[0])
            y_r.append(d[1])

        for (i, x) in enumerate(x_r):
            if i == 0:
                text += f"{x}"
            else:
                text += f" {x}"

        text += "];\ny = ["
        for (i, y) in enumerate(y_r):
            if i == 0:
                text += f"{y}"
            else:
                text += f" {y}"

        text += "];"

        return text

    elif mode == CopyOptions.POINTS_PYTHON:
        text = "x = ["
        x_r = []
        y_r = []

        for d in pts:
            x_r.append(d[0])
            y_r.append(d[1])

        for (i, x) in enumerate(x_r):
            if i == 0:
                text += f"{x}"
            else:
                text += f", {x}"

        text += "];\ny = ["
        for (i, y) in enumerate(y_r):
            if i == 0:
                text += f"{y}"
            else:
                text += f", {y}"

        text += "];"

        return text

    elif mode == CopyOptions.POINTS_NUMPY:
        text = "pts = np.array([["
        x_r = []
        y_r = []

        for d in pts:
            x_r.append(d[0])
            y_r.append(d[1])

        for (i, x) in enumerate(x_r):
            if i == 0:
                text += f"{x}"
            else:
                text += f", {x}"

        text += "], ["
        for (i, y) in enumerate(y_r):
            if i == 0:
                text += f"{y}"
            else:
                text += f", {y}"

        text += "]])"

        return text

    elif mode == CopyOptions.POINTS_CSV:
        text = "x, y\n"

        for d in pts:
            text += f"{d[0]}, {d[1]}\n"

        return text

    elif mode == CopyOptions.COEFFS_MATLAB:
        text = "["
        for (i, coef) in enumerate(coefs):
            if i == 0:
                text += f"{coef}"
            else:
                text += f" {coef}"
        text += "]"

        return text

    elif mode == CopyOptions.COEFFS_PYTHON:
        text = "["
        for (i, coef) in enumerate(coefs):
            if i == 0:
                text += f"{coef}"
            else:
                text += f", {coef}"
        text += "]"

        return text

    elif mode == CopyOptions.COEFFS_NUMPY:
        text = "np.array(["
        for (i, coef) in enumerate(coefs):
            if i == 0:
                text += f"{coef}"
            else:
                text += f", {coef}"
        text += "])"

        return text

    elif mode == CopyOptions.POLY1D:
        text = "p = np.poly1d(["
        for (i, coef) in enumerate(coefs):
            if i == 0:
                text += f"{coef}"
            else:
                text += f", {coef}"
        text += "])"

        return text

    else:
        return None

