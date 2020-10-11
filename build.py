from zipfile import ZipFile, ZIP_DEFLATED
import PyInstaller.__main__
import shutil
import os

VER = "2.2"

# Build the executable
PyInstaller.__main__.run([
    '--name=CurveFinder',
    '--onefile',
    '--noconsole',
    '--icon=resources/icon.ico',
    'curvefinder.py'
])

# Create the distribution zip file
with ZipFile('curvefinder_v{0:s}.zip'.format(VER), 'w', compression=ZIP_DEFLATED, compresslevel=9) as zipfile:
    zipfile.write('dist/CurveFinder.exe', 'CurveFinder.exe')
    zipfile.write('resources/placeholder.png', 'data/placeholder.png')
    zipfile.write('resources/icon.png', 'data/icon.png')

# Remove dist and build folder
shutil.rmtree('dist/')
shutil.rmtree('build/')
os.remove('CurveFinder.spec')
