@echo off
setlocal

:: ============================================================
:: LoRA Dataset Curator - Unified Launcher
:: ============================================================
:: Erstellt die venv, installiert Abhaengigkeiten, startet UI.
:: Einfach doppelklicken.
:: ============================================================

cd /d "%~dp0"

echo.
echo ========================================
echo   LoRA Dataset Curator
echo ========================================
echo.

:: --- Schritt 1: Virtuelle Umgebung ---
if exist "curator_env\Scripts\activate.bat" (
    echo [1/3] Virtuelle Umgebung vorhanden.
) else (
    echo [1/3] Erstelle virtuelle Umgebung...
    py -3.10 -m venv curator_env
    if errorlevel 1 goto :err_python
    echo       Erstellt.
)

call curator_env\Scripts\activate.bat

:: --- Schritt 2: Abhaengigkeiten ---
if exist "curator_env\_install_done.marker" (
    echo [2/3] Abhaengigkeiten bereits installiert.
) else (
    echo [2/3] Installiere Abhaengigkeiten...
    echo       Das kann einige Minuten dauern.
    echo.
    call :do_install
    if errorlevel 1 goto :err_install
    echo done > "curator_env\_install_done.marker"
    echo.
    echo       Installation abgeschlossen.
)

:: --- Schritt 3: UI starten ---
if not exist "dataset_curator_ui.py" goto :err_missing

echo [3/3] Starte UI...
echo.
echo ========================================
echo   UI startet im Browser automatisch.
echo   Port ist ggf. nicht 7860, wenn belegt.
echo   Zum Beenden: Ctrl+C hier druecken.
echo ========================================
echo.

:run_ui
python dataset_curator_ui.py

:: Exit code 5 = UI requested restart (e.g. after language change)
if %errorlevel%==5 (
    echo.
    echo [UI] Neustart angefordert...
    timeout /t 1 /nobreak >nul
    goto :run_ui
)

pause
exit /b 0


:: ============================================================
:: Installations-Subroutine
:: (Liegt ausserhalb aller if-Bloecke, damit Sonderzeichen
::  wie > = und Klammern keine Probleme machen.)
:: ============================================================
:do_install

echo       - pip upgrade...
python -m pip install --upgrade pip setuptools wheel

echo       - Basis-Pakete...
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
:: Fehlerbehandlung
:: ============================================================
:err_python
echo.
echo FEHLER: Python 3.10 nicht gefunden.
echo Bitte installieren: https://www.python.org/downloads/
pause
exit /b 1

:err_install
echo.
echo FEHLER bei der Installation. Siehe Ausgabe oben.
pause
exit /b 1

:err_missing
echo.
echo FEHLER: dataset_curator_ui.py nicht gefunden.
echo Bitte folgende Dateien hierher kopieren:
echo   - dataset_curator_ui.py
echo   - dataset_curator_v2.py
echo   - video_Processor.py
pause
exit /b 1
