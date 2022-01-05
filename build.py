from zipfile import ZipFile, ZIP_DEFLATED
from QCurveFinder.constants import VER
import PyInstaller.__main__
import argparse
import shutil
import os
####################################################

parser = argparse.ArgumentParser(description='Script to build the executable for the curvefinder software.')
parser.add_argument('-b', '--build', action="store_true", default=False, help="Build the app. [Default : False]")
parser.add_argument('-z', '--zip', action="store_true", default=False, help="Package the app in a zip file. "
                                                                            "[Default : False]")
parser.add_argument('-co', '--copy', action="store_true", default=False, help="Copy the app in this directory. "
                                                                              "[Default : False]")
parser.add_argument('-cl', '--clean', action="store_true", default=False, help="Clean the current directory. "
                                                                               "[Default : False]")
args = parser.parse_args()
####################################################

# Build the executable
if args.build:
    PyInstaller.__main__.run([
        '--name=CurveFinder',
        '--onefile',
        '--noconsole',
        '--icon=resources/icon.ico',
        '--add-data=resources/icon.ico;data',
        '--add-data=resources/placeholder.png;data',
        '--splash=resources/splash.png',
        'curvefinder.py'
    ])

# Copy the exe file from ./dist/ to current ./
if args.copy:
    if os.path.exists('CurveFinder.exe'):
        os.remove('CurveFinder.exe')
    shutil.copyfile('dist/CurveFinder.exe', 'CurveFinder.exe')

# Create the distribution zip file
if args.zip:
    filename = f'curvefinder_v{VER}.zip'
    if os.path.exists(filename):
        os.remove(filename)
    with ZipFile(filename, 'w', compression=ZIP_DEFLATED, compresslevel=9) as zipfile:
        zipfile.write('dist/CurveFinder.exe', 'CurveFinder.exe')

# Remove dist and build folder
if args.clean:
    shutil.rmtree('dist/')
    shutil.rmtree('build/')
    os.remove('CurveFinder.spec')
