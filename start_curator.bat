@echo off
setlocal

:: ============================================================
:: LoRA Dataset Curator - Unified Launcher
:: ============================================================
:: Creates the venv, installs dependencies, and starts the UI.
:: Just double-click.
::
:: Marker version (bump on dependency changes):
::   v1: initial
::   v2: switched onnxruntime -> onnxruntime-gpu for ArcFace
::       identity check on CUDA. Auto-migrates from v1.
:: ============================================================

cd /d "%~dp0"

echo.
echo ========================================
echo   LoRA Dataset Curator
echo ========================================
echo.

:: --- Step 1: Virtual environment ---
if exist "curator_env\Scripts\activate.bat" (
    echo Step 1 of 3: Virtual environment found
) else (
    echo Step 1 of 3: Creating virtual environment
    py -3.10 -m venv curator_env
    if errorlevel 1 goto err_python
    echo       Created
)

call curator_env\Scripts\activate.bat

:: --- Step 2: Dependencies ---
if exist "curator_env\_install_done.marker_v2" (
    echo Step 2 of 3: Dependencies already installed
) else (
    if exist "curator_env\_install_done.marker" (
        echo Step 2 of 3: Migrating from v1 install
        echo.
        call :do_migrate_v1_to_v2
        if errorlevel 1 goto err_install
    ) else (
        echo Step 2 of 3: Installing dependencies
        echo       This may take a few minutes
        echo.
        call :do_install
        if errorlevel 1 goto err_install
    )
    echo done > "curator_env\_install_done.marker_v2"
    echo.
    echo       Installation complete
)

:: --- Step 3: Start UI ---
if not exist "dataset_curator_ui.py" goto err_missing

echo Step 3 of 3: Starting UI
echo.
echo ========================================
echo   The UI will open automatically in your browser
echo   The port may not be 7860 if it is already in use
echo   To stop it, press Ctrl+C here
echo ========================================
echo.

:run_ui
python dataset_curator_ui.py

:: Exit code 5 = UI requested restart (e.g. after language change)
if %errorlevel%==5 (
    echo.
    echo UI restart requested
    timeout /t 1 /nobreak >nul
    goto run_ui
)

pause
exit /b 0


:: ============================================================
:: Full installation subroutine
:: ============================================================
:do_install

echo       - pip upgrade
python -m pip install --upgrade pip setuptools wheel

echo       - Base packages
python -m pip install requests pillow numpy scipy

echo       - MediaPipe
python -m pip install mediapipe==0.10.33

echo       - PyTorch + CUDA
python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130

echo       - OpenCLIP
python -m pip install open_clip_torch

echo       - OpenCV, InsightFace, scikit-learn
python -m pip install opencv-python insightface scikit-learn

echo       - ONNX Runtime (GPU)
python -m pip install onnxruntime-gpu

echo       - Gradio UI
python -m pip install gradio

exit /b 0


:: ============================================================
:: Migration subroutine: v1 install (CPU onnxruntime) -> v2 (GPU)
:: ============================================================
:do_migrate_v1_to_v2

echo       - Removing CPU-only onnxruntime
python -m pip uninstall -y onnxruntime

echo       - Installing GPU onnxruntime
python -m pip install onnxruntime-gpu

echo       - Verifying insightface is up to date
python -m pip install --upgrade insightface

exit /b 0


:: ============================================================
:: Error handling
:: ============================================================
:err_python
echo.
echo ERROR: Python 3.10 not found
echo Please install it from: https://www.python.org/downloads/
pause
exit /b 1

:err_install
echo.
echo ERROR during installation. See output above
pause
exit /b 1

:err_missing
echo.
echo ERROR: dataset_curator_ui.py not found
echo Please copy the following files here:
echo   - dataset_curator_ui.py
echo   - dataset_curator_v2.py
echo   - video_Processor.py
pause
exit /b 1
