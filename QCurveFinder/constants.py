from enum import IntEnum
import os


class AppState(IntEnum):
    INITIAL = 0
    STARTED = 1
    COORD_ALL_SELECTED = 2
    FILTER_CHOICE = 3
    EDGE_SELECTION = 4
    EQUATION_IMAGE = 5
    EQUATION_PLOT = 6


VER = "2.3"
AUTHOR = "Bruno-Pier Busque"

APP_W = 1500
APP_H = 750

MAX_IMG_W = 1200
MAX_IMG_H = 730

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/"))
ICON_PATH = os.path.join(DATA_PATH, "icon.ico")
PH_IMAGE_PATH = os.path.join(DATA_PATH, "placeholder.png")
TEMP_PATH = os.path.join(DATA_PATH, "temp/")
ORIG_IMG = os.path.join(TEMP_PATH, "original_img.png")
COOR_IMG = os.path.join(TEMP_PATH, "coordinate_img.png")
ROTA_IMG = os.path.join(TEMP_PATH, "rotated_img.png")
CONT_IMG = os.path.join(TEMP_PATH, "contoured_img.png")
CTMK_IMG = os.path.join(TEMP_PATH, "masked_contoured_img.png")
SELE_IMG = os.path.join(TEMP_PATH, "selected_img.png")
PLOT_IMG = os.path.join(TEMP_PATH, "plotted_img.png")

INITIAL_TEXT = "Choose a picture of a graph and press `Start`"
STARTED_TEXT = "1) Click on 2 points for each axis in this order:\n" \
               "    **X1 -> X2 -> Y1 -> Y2**\n" \
               "2) Enter the coordinates in the box below."
FILTER_CHOICE_TEXT = "Adjust the thresholding so that the curve you want to extract " \
                     "is clearly visible and free of close obstacles.\n\n" \
                     "When done, press `Next`"
EDGE_SELECTION_TEXT = "Press and hold over the curve you want to extract. All the " \
                      '"painted" multi-colored points will be extracted.\n\n' \
                      "When you selected all the curve, press `Next` to extract " \
                      "the data points."
EQUATION_TEXT = ""
