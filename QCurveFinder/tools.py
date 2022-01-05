from typing import Union, List
from numpy import ndarray
from .constants import *


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

