PROJECT UAS - DETEKSI WARNA OBJEK REAL-TIME

A. PERSIAPAN YANG DISARANKAN
- Windows 10/11 64-bit
- Python 3.14.6 64-bit
- Webcam internal atau eksternal
- VS Code atau Command Prompt/PowerShell

B. MEMBUAT VIRTUAL ENVIRONMENT
Buka Terminal di folder project, lalu jalankan:

Windows Command Prompt:
    py -3.14 -m venv .venv
    .venv\Scripts\activate.bat

Windows PowerShell:
    py -3.14 -m venv .venv
    .\.venv\Scripts\Activate.ps1

Jika PowerShell menolak aktivasi:
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    .\.venv\Scripts\Activate.ps1

C. INSTALASI LIBRARY
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

D. CEK VERSI
    python --version
    python -c "import cv2, numpy; print('OpenCV:', cv2.__version__); print('NumPy:', numpy.__version__)"

Versi yang diharapkan:
- Python 3.14.6
- OpenCV 4.13.0
- NumPy 2.3.5

E. UJI TANPA WEBCAM
    python deteksi_warna_realtime.py --self-test

Hasil sukses akan menampilkan tulisan SELF-TEST BERHASIL dan membuat:
- output/self_test_result.jpg
- output/self_test_mask.png

F. MENJALANKAN WEBCAM
    python deteksi_warna_realtime.py

G. MENJALANKAN MODE DEMO TANPA WEBCAM
    python deteksi_warna_realtime.py --demo

H. MEMAKAI VIDEO SENDIRI
Simpan video ke folder data_uji, misalnya video_uji.mp4, lalu jalankan:
    python deteksi_warna_realtime.py --source "data_uji/video_uji.mp4"

I. MEMAKAI KAMERA KEDUA
    python deteksi_warna_realtime.py --source 1

J. KONTROL PROGRAM
- Tombol 1: Merah
- Tombol 2: Jingga
- Tombol 3: Kuning
- Tombol 4: Hijau
- Tombol 5: Biru
- Tombol 6: Ungu
- Klik kiri pada warna objek: kalibrasi HSV otomatis
- Tombol S: simpan screenshot
- Tombol Q atau Esc: keluar

K. DATA UJI YANG WAJIB DITAMBAHKAN KELOMPOK
- Minimal satu video webcam real-time milik kelompok, atau dokumentasi beberapa skenario uji.
- Simpan video sendiri di folder data_uji.
- Simpan screenshot hasil pengujian di folder output dengan tombol S.
- Jangan hanya memakai gambar self-test karena gambar tersebut merupakan data sintetis.

L. CATATAN PENTING
- Jangan memasang opencv-python-headless karena program memakai cv2.imshow.
- Tutup Zoom, Google Meet, OBS, Camera, atau aplikasi lain yang memakai webcam.
- Gunakan objek berwarna pekat dan pencahayaan cukup.
- Jika objek kecil tidak terdeteksi, turunkan --min-area, contoh:
    python deteksi_warna_realtime.py --min-area 300
- Jika noise terlalu banyak, naikkan --min-area, contoh:
    python deteksi_warna_realtime.py --min-area 1500

M. CARA PALING MUDAH DI WINDOWS
1. Klik dua kali SETUP_DAN_TEST_WINDOWS.bat.
2. Tunggu sampai muncul SETUP DAN SELF-TEST BERHASIL.
3. Klik dua kali RUN_WEBCAM.bat.
4. Jika webcam tidak tersedia, klik RUN_DEMO_TANPA_WEBCAM.bat.
