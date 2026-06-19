@echo off
REM Setup script for NTFS Timestomping Detector - Bundled Python Environment
REM This script sets up a portable Python 3 runtime with all required dependencies

echo ========================================
echo NTFS Timestomping Detector Setup
echo ========================================
echo.

REM Get the directory where this script is located
set "MODULE_DIR=%~dp0"
set "PYTHON_RUNTIME_DIR=%MODULE_DIR%PythonRuntime"
set "EXTERNAL_PROCESSOR_DIR=%MODULE_DIR%ExternalProcessor"

echo Module directory: %MODULE_DIR%
echo Python runtime will be created at: %PYTHON_RUNTIME_DIR%
echo.

REM Check if PythonRuntime already exists
if exist "%PYTHON_RUNTIME_DIR%" (
    echo WARNING: PythonRuntime folder already exists!
    echo.
    choice /C YN /M "Do you want to rebuild it? (This will delete the existing runtime)"
    if errorlevel 2 goto :skip_download
    if errorlevel 1 (
        echo Removing existing PythonRuntime...
        rd /s /q "%PYTHON_RUNTIME_DIR%"
    )
)

:skip_download

REM Create PythonRuntime directory
if not exist "%PYTHON_RUNTIME_DIR%" (
    mkdir "%PYTHON_RUNTIME_DIR%"
)

echo.
echo ========================================
echo Step 1: Download Python Embeddable Package
echo ========================================
echo.
echo Please download Python 3.9 Embeddable Package:
echo URL: https://www.python.org/ftp/python/3.9.13/python-3.9.13-embed-amd64.zip
echo.
echo After downloading:
echo 1. Extract the contents to: %PYTHON_RUNTIME_DIR%
echo 2. Press any key to continue...
pause >nul

REM Verify python.exe exists
if not exist "%PYTHON_RUNTIME_DIR%\python.exe" (
    echo ERROR: python.exe not found in %PYTHON_RUNTIME_DIR%
    echo Please extract the Python embeddable package and try again.
    pause
    exit /b 1
)

echo Found python.exe
echo.

echo ========================================
echo Step 2: Configure Python for pip
echo ========================================
echo.

REM Find the ._pth file
for %%f in ("%PYTHON_RUNTIME_DIR%\python*._pth") do (
    set "PTH_FILE=%%f"
    echo Found PTH file: %%f
)

if not defined PTH_FILE (
    echo ERROR: Could not find python*._pth file
    pause
    exit /b 1
)

REM Enable site-packages by uncommenting import site
echo Enabling site-packages...
powershell -Command "(Get-Content '%PTH_FILE%') -replace '#import site', 'import site' | Set-Content '%PTH_FILE%'"
echo Done.
echo.

echo ========================================
echo Step 3: Download and Install pip
echo ========================================
echo.

set "GET_PIP=%PYTHON_RUNTIME_DIR%\get-pip.py"

REM Download get-pip.py
echo Downloading get-pip.py...
powershell -Command "Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile '%GET_PIP%'"

if not exist "%GET_PIP%" (
    echo ERROR: Failed to download get-pip.py
    echo Please download manually from https://bootstrap.pypa.io/get-pip.py
    echo and save it to: %GET_PIP%
    pause
    exit /b 1
)

echo Installing pip...
"%PYTHON_RUNTIME_DIR%\python.exe" "%GET_PIP%"

if errorlevel 1 (
    echo ERROR: Failed to install pip
    pause
    exit /b 1
)

echo pip installed successfully.
echo.

echo ========================================
echo Step 4: Install Required Dependencies
echo ========================================
echo.

echo Installing pandas (this may take a few minutes)...
"%PYTHON_RUNTIME_DIR%\python.exe" -m pip install pandas

echo.
echo Installing numpy...
"%PYTHON_RUNTIME_DIR%\python.exe" -m pip install numpy

echo.
echo Installing pytz...
"%PYTHON_RUNTIME_DIR%\python.exe" -m pip install pytz

echo.
echo Installing python-dateutil...
"%PYTHON_RUNTIME_DIR%\python.exe" -m pip install python-dateutil

echo.
echo Installing scikit-learn...
"%PYTHON_RUNTIME_DIR%\python.exe" -m pip install scikit-learn

echo.
echo Installing lightgbm...
"%PYTHON_RUNTIME_DIR%\python.exe" -m pip install lightgbm

echo.
echo Installing xgboost...
"%PYTHON_RUNTIME_DIR%\python.exe" -m pip install xgboost

echo.
echo Installing imbalanced-learn...
"%PYTHON_RUNTIME_DIR%\python.exe" -m pip install imbalanced-learn

echo.
echo Installing optuna...
"%PYTHON_RUNTIME_DIR%\python.exe" -m pip install optuna

echo.
echo Installing joblib...
"%PYTHON_RUNTIME_DIR%\python.exe" -m pip install joblib

echo.
echo Installing matplotlib...
"%PYTHON_RUNTIME_DIR%\python.exe" -m pip install matplotlib

echo.
echo Installing seaborn...
"%PYTHON_RUNTIME_DIR%\python.exe" -m pip install seaborn

echo.
echo Installing construct (required by dfir_ntfs)...
"%PYTHON_RUNTIME_DIR%\python.exe" -m pip install construct

echo.
echo All dependencies installed.
echo.
echo.

echo ========================================
echo Step 5: Verify Installation
echo ========================================
echo.

REM Create a test script
set "TEST_SCRIPT=%MODULE_DIR%test_bundled_python.py"
(
echo import sys
echo import os
echo.
echo # Add ExternalProcessor to path for dfir_ntfs testing
echo module_dir = os.path.dirname^(os.path.abspath^(__file__^)^)
echo sys.path.insert^(0, os.path.join^(module_dir, 'ExternalProcessor'^)^)
echo.
echo print^("="*80^)
echo print^("Python Environment Verification Test"^)
echo print^("="*80^)
echo print^("Python version:", sys.version^)
echo print^("Python executable:", sys.executable^)
echo print^("="*80^)
echo print^()
echo.
echo print^("Testing core data processing libraries..."^)
echo try:
echo     import pandas as pd
echo     print^("  ✓ pandas:", pd.__version__^)
echo except ImportError as e:
echo     print^("  ✗ pandas import failed:", e^)
echo     sys.exit^(1^)
echo.
echo try:
echo     import numpy as np
echo     print^("  ✓ numpy:", np.__version__^)
echo except ImportError as e:
echo     print^("  ✗ numpy import failed:", e^)
echo     sys.exit^(1^)
echo.
echo print^()
echo print^("Testing machine learning libraries..."^)
echo try:
echo     import sklearn
echo     print^("  ✓ scikit-learn:", sklearn.__version__^)
echo except ImportError as e:
echo     print^("  ✗ scikit-learn import failed:", e^)
echo     sys.exit^(1^)
echo.
echo try:
echo     import lightgbm as lgb
echo     print^("  ✓ lightgbm:", lgb.__version__^)
echo except ImportError as e:
echo     print^("  ✗ lightgbm import failed:", e^)
echo     sys.exit^(1^)
echo.
echo try:
echo     import xgboost as xgb
echo     print^("  ✓ xgboost:", xgb.__version__^)
echo except ImportError as e:
echo     print^("  ✗ xgboost import failed:", e^)
echo     sys.exit^(1^)
echo.
echo try:
echo     import imblearn
echo     print^("  ✓ imbalanced-learn:", imblearn.__version__^)
echo except ImportError as e:
echo     print^("  ✗ imbalanced-learn import failed:", e^)
echo     sys.exit^(1^)
echo.
echo try:
echo     import optuna
echo     print^("  ✓ optuna:", optuna.__version__^)
echo except ImportError as e:
echo     print^("  ✗ optuna import failed:", e^)
echo     sys.exit^(1^)
echo.
echo try:
echo     import joblib
echo     print^("  ✓ joblib:", joblib.__version__^)
echo except ImportError as e:
echo     print^("  ✗ joblib import failed:", e^)
echo     sys.exit^(1^)
echo.
echo print^()
echo print^("Testing visualization libraries..."^)
echo try:
echo     import matplotlib
echo     print^("  ✓ matplotlib:", matplotlib.__version__^)
echo except ImportError as e:
echo     print^("  ✗ matplotlib import failed:", e^)
echo     sys.exit^(1^)
echo.
echo try:
echo     import seaborn as sns
echo     print^("  ✓ seaborn:", sns.__version__^)
echo except ImportError as e:
echo     print^("  ✗ seaborn import failed:", e^)
echo     sys.exit^(1^)
echo.
echo print^()
echo print^("Testing parsing library dependencies..."^)
echo try:
echo     import construct
echo     print^("  ✓ construct:", construct.__version__^)
echo except ImportError as e:
echo     print^("  ✗ construct import failed:", e^)
echo     sys.exit^(1^)
echo.
echo print^()
echo print^("Testing bundled dfir_ntfs library..."^)
echo try:
echo     from Parser.ThirdParty.dfir_ntfs import MFT
echo     print^("  ✓ dfir_ntfs.MFT imported successfully"^)
echo except ImportError as e:
echo     print^("  ✗ dfir_ntfs.MFT import failed:", e^)
echo     print^("     Make sure dfir_ntfs folder exists in ExternalProcessor/Parser/ThirdParty/"^)
echo     sys.exit^(1^)
echo.
echo try:
echo     from Parser.ThirdParty.dfir_ntfs.USN import ChangeJournalParser
echo     print^("  ✓ dfir_ntfs.USN.ChangeJournalParser imported successfully"^)
echo except ImportError as e:
echo     print^("  ✗ dfir_ntfs.USN import failed:", e^)
echo     sys.exit^(1^)
echo.
echo try:
echo     from Parser.ThirdParty.dfir_ntfs.LogFile import LogFileParser as LFP
echo     print^("  ✓ dfir_ntfs.LogFile.LogFileParser imported successfully"^)
echo except ImportError as e:
echo     print^("  ✗ dfir_ntfs.LogFile import failed:", e^)
echo     sys.exit^(1^)
echo.
echo print^()
echo print^("="*80^)
echo print^("SUCCESS! All dependencies verified and ready to use!"^)
echo print^("="*80^)
echo print^()
echo print^("Summary:"^)
echo print^("  - Core libraries: pandas, numpy"^)
echo print^("  - ML libraries: scikit-learn, lightgbm, xgboost, imbalanced-learn, optuna"^)
echo print^("  - Utilities: joblib, construct"^)
echo print^("  - Visualization: matplotlib, seaborn"^)
echo print^("  - NTFS parsing: dfir_ntfs ^(bundled^)"^)
) > "%TEST_SCRIPT%"

echo Running verification test...
echo.
"%PYTHON_RUNTIME_DIR%\python.exe" "%TEST_SCRIPT%"

if errorlevel 1 (
    echo.
    echo ERROR: Verification failed!
    echo Please check the error messages above.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Step 6: Cleanup
echo ========================================
echo.

echo Removing temporary files...
del "%GET_PIP%"
del "%TEST_SCRIPT%"

echo Removing pip cache to save space...
rd /s /q "%PYTHON_RUNTIME_DIR%\Lib\site-packages\pip\_vendor\" 2>nul

echo Removing __pycache__ directories...
for /d /r "%PYTHON_RUNTIME_DIR%\Lib" %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"

echo Cleanup complete.
echo.

echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Your bundled Python environment is ready at:
echo %PYTHON_RUNTIME_DIR%
echo.
echo Next steps:
echo 1. Copy your module to Autopsy's python_modules directory
echo 2. Restart Autopsy
echo 3. The module will use the bundled Python automatically
echo.
echo Module size information:
dir "%PYTHON_RUNTIME_DIR%" | find "File(s)"
echo.
pause