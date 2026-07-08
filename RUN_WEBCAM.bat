@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment belum dibuat.
    echo Jalankan SETUP_DAN_TEST_WINDOWS.bat terlebih dahulu.
    pause
    exit /b 1
)
call .venv\Scripts\activate.bat
python deteksi_warna_realtime.py
pause
