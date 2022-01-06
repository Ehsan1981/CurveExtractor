from PyQt5.QtWidgets import QApplication
from QCurveFinder.application import CurveFinder

try:
    import pyi_splash
    splash = True
except ModuleNotFoundError:
    splash = False

from random import seed
import sys


seed(123456)
app = QApplication([])
window = CurveFinder()
if splash:
    pyi_splash.close()
window.show()
sys.exit(app.exec_())
