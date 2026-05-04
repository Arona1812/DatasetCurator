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
::   v3: removed unused torchaudio dependency. Auto-handled by
::       the verify check; existing installs simply have an
::       extra package that no longer hurts.
::
:: Important: the marker is only a cache hint. The launcher always performs
:: real Python import checks before starting the UI and repairs missing
:: packages automatically.
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
echo Step 2 of 3: Checking dependencies
call :verify_dependencies
if errorlevel 1 (
    echo       Missing or broken core dependencies detected
    echo.
    if exist "curator_env\_install_done.marker" (
        echo Step 2 of 3: Migrating from v1 install
        echo.
        call :do_migrate_v1_to_v2
        if errorlevel 1 goto err_install
    ) else (
        echo Step 2 of 3: Installing/repairing dependencies
        echo       This may take a few minutes
        echo.
        call :do_install
        if errorlevel 1 goto err_install
    )

    echo.
    echo       Re-checking core dependencies
    call :verify_dependencies
    if errorlevel 1 goto err_verify

    echo done > "curator_env\_install_done.marker_v2"
    echo.
    echo       Dependencies ready
) else (
    if exist "curator_env\_install_done.marker_v2" (
        echo       Dependencies verified
    ) else (
        echo done > "curator_env\_install_done.marker_v2"
        echo       Dependencies verified and marker created
    )
)

echo       Checking optional InsightFace support
call :verify_optional_dependencies
if errorlevel 1 (
    echo       Optional InsightFace support is missing. Trying repair...
    call :do_install_optional
    call :verify_optional_dependencies
    if errorlevel 1 (
        echo.
        echo WARNING: InsightFace is still not available.
        echo          The UI and image curator can start, but ArcFace identity
        echo          check and Video Processor face recognition will be disabled
        echo          or fail until InsightFace is installed.
        echo          On Windows, InsightFace may require Microsoft C++ Build Tools:
        echo          https://visualstudio.microsoft.com/visual-cpp-build-tools/
        echo.
    )
)

if /I "%~1"=="--check-only" (
    echo Check-only mode complete. UI was not started.
    exit /b 0
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
:: Dependency verification subroutine
:: ============================================================
:verify_dependencies

python -c "import importlib.util, sys; mods=['requests','PIL','numpy','scipy','mediapipe','torch','torchvision','open_clip','cv2','onnxruntime','sklearn','gradio']; missing=[m for m in mods if importlib.util.find_spec(m) is None]; print('      Python ' + sys.version.split()[0]); print('      Missing core: ' + (', '.join(missing) if missing else 'none')); sys.exit(1 if missing else 0)"

exit /b %errorlevel%


:: ============================================================
:: Optional dependency verification subroutine
:: ============================================================
:verify_optional_dependencies

python -c "import importlib.util, sys; mods=['insightface']; missing=[m for m in mods if importlib.util.find_spec(m) is None]; print('      Missing optional: ' + (', '.join(missing) if missing else 'none')); sys.exit(1 if missing else 0)"

exit /b %errorlevel%


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
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu130

echo       - OpenCLIP
python -m pip install open_clip_torch

echo       - OpenCV, scikit-learn
python -m pip install opencv-python scikit-learn

echo       - ONNX Runtime (GPU)
python -m pip install onnxruntime-gpu

echo       - Gradio UI
python -m pip install gradio

call :do_install_optional

exit /b 0


:: ============================================================
:: Optional installation subroutine
:: ============================================================
:do_install_optional

echo       - Optional InsightFace
python -m pip install insightface
if errorlevel 1 (
    echo       InsightFace installation failed; continuing without optional support
    exit /b 0
)

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

:err_verify
echo.
echo ERROR: Dependencies are still missing after installation/repair.
echo Please check the output above. Common fixes:
echo   curator_env\Scripts\python.exe -m pip install scikit-learn
echo   curator_env\Scripts\python.exe -m pip install opencv-python
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
