from zipfile import ZipFile, ZIP_DEFLATED
import PyInstaller.__main__
from curvefinder import VER
import shutil
import os


# Build the executable
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

# Create the distribution zip file
with ZipFile(f'curvefinder_v{VER}.zip', 'w', compression=ZIP_DEFLATED, compresslevel=9) as zipfile:
    zipfile.write('dist/CurveFinder.exe', 'CurveFinder.exe')

# Remove dist and build folder
shutil.rmtree('dist/')
shutil.rmtree('build/')
os.remove('CurveFinder.spec')
