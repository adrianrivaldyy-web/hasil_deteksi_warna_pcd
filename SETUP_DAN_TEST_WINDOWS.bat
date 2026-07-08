@echo off
setlocal
cd /d "%~dp0"

echo ================================================
echo SETUP PROJECT DETEKSI WARNA REAL-TIME
echo ================================================

where py >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python Launcher ^(py^) tidak ditemukan.
    echo Instal Python 3.14.6 64-bit dari python.org dan aktifkan Python Launcher.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Membuat virtual environment dengan Python 3.14...
    py -3.14 -m venv .venv >nul 2>nul
    if errorlevel 1 (
        echo Python 3.14 tidak ditemukan. Mencoba versi Python default...
        py -m venv .venv
    )
    if errorlevel 1 (
        echo ERROR: virtual environment gagal dibuat.
        echo Cek instalasi dengan perintah: py --version
        pause
        exit /b 1
    )
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
if errorlevel 1 goto :error
python -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo.
echo Versi yang digunakan:
python --version
python -c "import cv2, numpy; print('OpenCV:', cv2.__version__); print('NumPy:', numpy.__version__)"
if errorlevel 1 goto :error

echo.
echo Menjalankan self-test...
python deteksi_warna_realtime.py --self-test
if errorlevel 1 goto :error

echo.
echo SETUP DAN SELF-TEST BERHASIL.
echo Jalankan RUN_WEBCAM.bat untuk membuka kamera.
pause
exit /b 0

:error
echo.
echo ERROR: setup atau self-test gagal. Baca pesan error di atas.
pause
exit /b 1
