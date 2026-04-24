@echo off
setlocal

:: ============================================================
:: LoRA Dataset Curator - Unified Launcher
:: ============================================================
:: Creates the venv, installs dependencies, and starts the UI.
:: Just double-click.
:: ============================================================

cd /d "%~dp0"

echo.
echo ========================================
echo   LoRA Dataset Curator
echo ========================================
echo.

:: --- Step 1: Virtual environment ---
if exist "curator_env\Scripts\activate.bat" (
    echo [1/3] Virtual environment found.
) else (
    echo [1/3] Creating virtual environment...
    py -3.10 -m venv curator_env
    if errorlevel 1 goto :err_python
    echo       Created.
)

call curator_env\Scripts\activate.bat

:: --- Step 2: Dependencies ---
if exist "curator_env\_install_done.marker" (
    echo [2/3] Dependencies already installed.
) else (
    echo [2/3] Installing dependencies...
    echo       This may take a few minutes.
    echo.
    call :do_install
    if errorlevel 1 goto :err_install
    echo done > "curator_env\_install_done.marker"
    echo.
    echo       Installation complete.
)

:: --- Step 3: Start UI ---
if not exist "dataset_curator_ui.py" goto :err_missing

echo [3/3] Starting UI...
echo.
echo ========================================
echo   The UI will open automatically in your browser.
echo   The port may not be 7860 if it is already in use.
echo   To stop it, press Ctrl+C here.
echo ========================================
echo.

:run_ui
python dataset_curator_ui.py

:: Exit code 5 = UI requested restart (e.g. after language change)
if %errorlevel%==5 (
    echo.
    echo [UI] Restart requested...
    timeout /t 1 /nobreak >nul
    goto :run_ui
)

pause
exit /b 0


:: ============================================================
:: Installation subroutine
:: (Placed outside all if blocks so that special characters
::  such as > = and parentheses do not cause issues.)
:: ============================================================
:do_install

echo       - pip upgrade...
python -m pip install --upgrade pip setuptools wheel

echo       - Base packages...
python -m pip install requests pillow numpy scipy

echo       - MediaPipe...
python -m pip install mediapipe==0.10.33

echo       - PyTorch + CUDA...
python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130

echo       - OpenCLIP...
python -m pip install open_clip_torch

echo       - OpenCV, InsightFace, scikit-learn...
python -m pip install opencv-python insightface onnxruntime scikit-learn

echo       - Gradio UI...
python -m pip install gradio

exit /b 0


:: ============================================================
:: Error handling
:: ============================================================
:err_python
echo.
echo ERROR: Python 3.10 not found.
echo Please install it from: https://www.python.org/downloads/
pause
exit /b 1

:err_install
echo.
echo ERROR during installation. See output above.
pause
exit /b 1

:err_missing
echo.
echo ERROR: dataset_curator_ui.py not found.
echo Please copy the following files here:
echo   - dataset_curator_ui.py
echo   - dataset_curator_v2.py
echo   - video_Processor.py
pause
exit /b 1
