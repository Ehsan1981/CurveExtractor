from QCurveFinder.constants import VER
import PyInstaller.__main__
import argparse
import platform
import shutil
import sys
import os
####################################################

parser = argparse.ArgumentParser(description='Script to build the executable for the curvefinder software.')
parser.add_argument('-b', '--build', action="store_true", default=False, help="Build the app. [Default : False]")
parser.add_argument('-cl', '--clean', action="store_true", default=False, help="Clean the current directory. "
                                                                               "[Default : False]")
args = parser.parse_args()
####################################################

# Build the executable
if args.build:
    arguments = [
        'curvefinder.py',
        '--onefile',
        '--noconsole',
        '--icon=resources/icon.ico'
    ]

    if platform.system() == "Windows":
        os_name = "win"
        separator = ';'
        splash = True
    elif platform.system() == "Linux":
        os_name = "linux"
        separator = ':'
        splash = True
    elif platform.system() == "Darwin":
        os_name = "macos"
        separator = ':'
        splash = False
    else:
        sys.exit(0)

    arguments.append(f'--name=CurveFinder_v{VER}_{os_name}')
    arguments.append(f'--add-data=resources/icon.ico{separator}data')
    arguments.append(f'--add-data=resources/placeholder.png{separator}data')

    if splash:
        arguments.append('--splash=resources/splash.png')

    PyInstaller.__main__.run(arguments)

# Remove dist and build folder
if args.clean:
    shutil.rmtree('dist/')
    shutil.rmtree('build/')
    os.remove('CurveFinder.spec')
