"""
Project UAS Pengolahan Citra Digital
Topik 5: Deteksi Warna Objek Real-Time menggunakan OpenCV

Fitur utama:
1. Deteksi satu warna target secara real-time dengan HSV masking.
2. Pilihan enam preset warna atau kalibrasi warna dengan klik mouse.
3. Reduksi noise memakai Gaussian blur, morphology opening, dan closing.
4. Bounding box, contour, centroid, luas, jumlah objek, dan FPS.
5. Simpan screenshot hasil dengan tombol S.
6. Mode demo dan self-test agar program dapat diuji tanpa webcam.

Kontrol saat program berjalan:
- 1: Merah
- 2: Jingga
- 3: Kuning
- 4: Hijau
- 5: Biru
- 6: Ungu
- Klik kiri pada objek: kalibrasi warna target dari titik yang dipilih
- S: simpan screenshot
- Q atau Esc: keluar
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import cv2
import numpy as np


WINDOW_NAME = "Deteksi Warna Objek Real-Time"
SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class HSVRange:
    """Menyimpan batas bawah dan atas HSV."""

    lower: tuple[int, int, int]
    upper: tuple[int, int, int]

    def as_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.array(self.lower, dtype=np.uint8),
            np.array(self.upper, dtype=np.uint8),
        )


# Rentang HSV dirancang cukup umum untuk kondisi pencahayaan normal.
# Warna merah memakai dua rentang karena hue merah berada di awal dan akhir skala Hue OpenCV.
COLOR_PRESETS: dict[str, list[HSVRange]] = {
    "Merah": [
        HSVRange((0, 100, 70), (10, 255, 255)),
        HSVRange((170, 100, 70), (179, 255, 255)),
    ],
    "Jingga": [HSVRange((11, 100, 80), (23, 255, 255))],
    "Kuning": [HSVRange((24, 90, 80), (35, 255, 255))],
    "Hijau": [HSVRange((36, 60, 50), (85, 255, 255))],
    "Biru": [HSVRange((90, 70, 50), (135, 255, 255))],
    "Ungu": [HSVRange((136, 60, 50), (169, 255, 255))],
}

KEY_TO_COLOR = {
    ord("1"): "Merah",
    ord("2"): "Jingga",
    ord("3"): "Kuning",
    ord("4"): "Hijau",
    ord("5"): "Biru",
    ord("6"): "Ungu",
}

# Warna garis bounding box dalam format BGR.
DRAW_COLORS: dict[str, tuple[int, int, int]] = {
    "Merah": (0, 0, 255),
    "Jingga": (0, 140, 255),
    "Kuning": (0, 255, 255),
    "Hijau": (0, 255, 0),
    "Biru": (255, 0, 0),
    "Ungu": (255, 0, 255),
    "Kustom": (255, 255, 255),
}


class SyntheticVideoSource:
    """Membuat video sintetis bergerak untuk demo tanpa webcam."""

    def __init__(self, width: int = 960, height: int = 540) -> None:
        self.width = width
        self.height = height
        self.start_time = time.perf_counter()
        self.opened = True

    def isOpened(self) -> bool:  # noqa: N802 - mengikuti nama method OpenCV
        return self.opened

    def read(self) -> tuple[bool, np.ndarray]:
        if not self.opened:
            return False, np.empty((0, 0, 3), dtype=np.uint8)

        t = time.perf_counter() - self.start_time
        frame = np.full((self.height, self.width, 3), 35, dtype=np.uint8)

        # Latar belakang gradasi agar demo lebih realistis.
        for y in range(self.height):
            value = int(35 + (y / max(1, self.height - 1)) * 30)
            frame[y, :, :] = (value, value, value)

        objects = [
            ("Merah", (0, 0, 255), 0.0, 105),
            ("Jingga", (0, 140, 255), 0.8, 205),
            ("Kuning", (0, 255, 255), 1.6, 305),
            ("Hijau", (0, 255, 0), 2.4, 405),
            ("Biru", (255, 0, 0), 3.2, 155),
            ("Ungu", (255, 0, 255), 4.0, 355),
        ]

        for index, (name, bgr, phase, base_y) in enumerate(objects):
            x = int(self.width / 2 + math.sin(t * 0.8 + phase) * (self.width * 0.34))
            y = int(base_y + math.cos(t * 1.1 + phase) * 32)
            radius = 34 + (index % 3) * 5
            cv2.circle(frame, (x, y), radius, bgr, -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), radius, (245, 245, 245), 2, cv2.LINE_AA)
            cv2.putText(
                frame,
                name,
                (x - 35, y - radius - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            frame,
            "MODE DEMO TANPA WEBCAM",
            (24, self.height - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (230, 230, 230),
            2,
            cv2.LINE_AA,
        )
        return True, frame

    def release(self) -> None:
        self.opened = False

    def set(self, _property_id: int, _value: float) -> bool:
        return True


VideoSource = Union[cv2.VideoCapture, SyntheticVideoSource]


def parse_source(value: str) -> Union[int, str]:
    """Mengubah input angka menjadi indeks kamera, selain itu dianggap sebagai path video."""
    stripped = value.strip()
    if stripped.lstrip("-").isdigit():
        return int(stripped)
    return stripped


def resize_letterbox(
    frame: np.ndarray, target_width: int, target_height: int
) -> np.ndarray:
    """Resize dengan rasio aspek tetap lalu menambahkan padding hitam."""
    source_height, source_width = frame.shape[:2]
    if source_width <= 0 or source_height <= 0:
        raise ValueError("Ukuran frame tidak valid.")

    scale = min(target_width / source_width, target_height / source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(
        frame, (resized_width, resized_height), interpolation=interpolation
    )

    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    offset_x = (target_width - resized_width) // 2
    offset_y = (target_height - resized_height) // 2
    canvas[
        offset_y : offset_y + resized_height,
        offset_x : offset_x + resized_width,
    ] = resized
    return canvas


def create_mask(hsv_frame: np.ndarray, ranges: Sequence[HSVRange]) -> np.ndarray:
    """Membuat satu mask gabungan dari satu atau beberapa rentang HSV."""
    mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
    for hsv_range in ranges:
        lower, upper = hsv_range.as_numpy()
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv_frame, lower, upper))
    return mask


def clean_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Menghilangkan noise kecil dan menutup lubang pada objek."""
    if kernel_size < 3:
        return mask
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
    return closed


def find_external_contours(mask: np.ndarray) -> list[np.ndarray]:
    """Kompatibel dengan return value findContours pada beberapa versi OpenCV."""
    result = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = result[0] if len(result) == 2 else result[1]
    return list(contours)


def custom_ranges_from_hsv(
    hue: int,
    saturation: int,
    value: int,
    hue_tolerance: int = 10,
    saturation_tolerance: int = 80,
    value_tolerance: int = 80,
) -> list[HSVRange]:
    """Membuat rentang HSV kustom dan menangani hue yang melewati 0 atau 179."""
    hue = int(np.clip(hue, 0, 179))
    saturation = int(np.clip(saturation, 0, 255))
    value = int(np.clip(value, 0, 255))

    low_s = max(45, saturation - saturation_tolerance)
    high_s = 255
    low_v = max(45, value - value_tolerance)
    high_v = 255

    low_h = hue - hue_tolerance
    high_h = hue + hue_tolerance

    if low_h < 0:
        return [
            HSVRange((0, low_s, low_v), (high_h, high_s, high_v)),
            HSVRange((180 + low_h, low_s, low_v), (179, high_s, high_v)),
        ]
    if high_h > 179:
        return [
            HSVRange((low_h, low_s, low_v), (179, high_s, high_v)),
            HSVRange((0, low_s, low_v), (high_h - 180, high_s, high_v)),
        ]
    return [HSVRange((low_h, low_s, low_v), (high_h, high_s, high_v))]


def put_text_with_background(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    font_scale: float = 0.55,
    foreground: tuple[int, int, int] = (255, 255, 255),
    background: tuple[int, int, int] = (20, 20, 20),
    thickness: int = 1,
) -> None:
    """Menulis teks dengan kotak latar agar tetap terbaca."""
    x, y = origin
    (width, height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    top_left = (max(0, x - 4), max(0, y - height - 5))
    bottom_right = (
        min(image.shape[1] - 1, x + width + 5),
        min(image.shape[0] - 1, y + baseline + 4),
    )
    cv2.rectangle(image, top_left, bottom_right, background, -1)
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        foreground,
        thickness,
        cv2.LINE_AA,
    )


def draw_detections(
    frame: np.ndarray,
    contours: Iterable[np.ndarray],
    target_name: str,
    min_area: float,
) -> tuple[np.ndarray, int]:
    """Menggambar contour, bounding box, centroid, dan luas objek."""
    annotated = frame.copy()
    base_name = target_name.split(" ")[0]
    line_color = DRAW_COLORS.get(base_name, DRAW_COLORS["Kustom"])

    valid_contours = [
        contour for contour in contours if cv2.contourArea(contour) >= min_area
    ]
    valid_contours.sort(key=cv2.contourArea, reverse=True)

    for index, contour in enumerate(valid_contours, start=1):
        area = cv2.contourArea(contour)
        x, y, width, height = cv2.boundingRect(contour)
        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"])
        else:
            center_x = x + width // 2
            center_y = y + height // 2

        cv2.drawContours(annotated, [contour], -1, line_color, 2)
        cv2.rectangle(annotated, (x, y), (x + width, y + height), line_color, 2)
        cv2.circle(annotated, (center_x, center_y), 5, (255, 255, 255), -1)
        cv2.line(
            annotated,
            (center_x - 9, center_y),
            (center_x + 9, center_y),
            (255, 255, 255),
            1,
        )
        cv2.line(
            annotated,
            (center_x, center_y - 9),
            (center_x, center_y + 9),
            (255, 255, 255),
            1,
        )

        label_y = max(22, y - 8)
        put_text_with_background(
            annotated,
            f"Objek {index} | Luas: {area:.0f} px",
            (x, label_y),
            font_scale=0.5,
            foreground=(255, 255, 255),
            background=line_color,
        )

    return annotated, len(valid_contours)


def add_dashboard_information(
    frame: np.ndarray,
    target_name: str,
    object_count: int,
    fps: float,
    message: str,
) -> np.ndarray:
    """Menambahkan judul, status, FPS, jumlah objek, dan bantuan tombol."""
    output = frame.copy()
    height, width = output.shape[:2]

    # Header transparan.
    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (width, 88), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.72, output, 0.28, 0, output)

    cv2.putText(
        output,
        "DETEKSI WARNA OBJEK REAL-TIME",
        (18, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        f"Target: {target_name} | Jumlah: {object_count} | FPS: {fps:.1f}",
        (18, 61),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.57,
        (220, 255, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        "1 Merah | 2 Jingga | 3 Kuning | 4 Hijau | 5 Biru | 6 Ungu | Klik: kalibrasi | S: simpan | Q: keluar",
        (18, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )

    if message:
        put_text_with_background(
            output,
            message,
            (18, height - 18),
            font_scale=0.5,
            foreground=(255, 255, 255),
            background=(25, 25, 25),
        )
    return output


def build_dashboard(
    annotated: np.ndarray,
    mask: np.ndarray,
    masked_result: np.ndarray,
) -> np.ndarray:
    """Menggabungkan hasil utama, mask, dan hasil masking dalam satu dashboard."""
    height, width = annotated.shape[:2]
    preview_height = max(140, int(height * 0.30))
    preview_width = width // 2

    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_preview = cv2.resize(mask_bgr, (preview_width, preview_height))
    result_preview = cv2.resize(
        masked_result, (width - preview_width, preview_height)
    )

    put_text_with_background(mask_preview, "MASK BINARY", (12, 27), 0.55)
    put_text_with_background(result_preview, "HASIL MASKING", (12, 27), 0.55)

    bottom = np.hstack((mask_preview, result_preview))
    return np.vstack((annotated, bottom))


def create_video_source(
    source: Union[int, str], width: int, height: int, demo: bool
) -> VideoSource:
    """Membuka webcam/video dengan fallback backend pada Windows."""
    if demo:
        return SyntheticVideoSource(width=width, height=height)

    if isinstance(source, int) and sys.platform.startswith("win"):
        capture = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(source)
    else:
        capture = cv2.VideoCapture(source)

    if capture.isOpened() and isinstance(source, int):
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def save_screenshot(image: np.ndarray, output_dir: Path) -> Path:
    """Menyimpan screenshot menggunakan timestamp unik."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = output_dir / f"hasil_deteksi_{timestamp}.jpg"
    if not cv2.imwrite(str(output_path), image):
        raise OSError(f"Gagal menyimpan screenshot ke: {output_path}")
    return output_path


def generate_self_test_image(width: int = 960, height: int = 540) -> np.ndarray:
    """Membuat citra uji berisi objek merah dan noise kecil."""
    frame = np.full((height, width, 3), 45, dtype=np.uint8)
    cv2.rectangle(frame, (130, 140), (360, 360), (0, 0, 255), -1)
    cv2.circle(frame, (650, 250), 105, (0, 0, 255), -1)
    cv2.circle(frame, (790, 410), 8, (0, 0, 255), -1)  # noise kecil
    cv2.rectangle(frame, (420, 90), (540, 190), (255, 0, 0), -1)
    return frame


def process_frame(
    frame: np.ndarray,
    hsv_ranges: Sequence[HSVRange],
    target_name: str,
    min_area: float,
    kernel_size: int,
    fps: float,
    message: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
    """Pipeline lengkap pengolahan satu frame."""
    # 1. Preprocessing: blur untuk mengurangi noise detail kecil.
    blurred = cv2.GaussianBlur(frame, (7, 7), 0)

    # 2. Konversi BGR ke HSV.
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # 3. Segmentasi warna dengan threshold rentang HSV.
    raw_mask = create_mask(hsv, hsv_ranges)

    # 4. Perbaikan mask memakai opening dan closing.
    mask = clean_mask(raw_mask, kernel_size=kernel_size)

    # 5. Contour detection dan pemfilteran luas objek.
    contours = find_external_contours(mask)
    annotated, object_count = draw_detections(
        frame, contours, target_name=target_name, min_area=min_area
    )

    # 6. Hasil masking untuk menampilkan hanya warna target.
    masked_result = cv2.bitwise_and(frame, frame, mask=mask)

    # 7. Informasi visual dan dashboard output.
    annotated = add_dashboard_information(
        annotated,
        target_name=target_name,
        object_count=object_count,
        fps=fps,
        message=message,
    )
    dashboard = build_dashboard(annotated, mask, masked_result)
    return dashboard, mask, hsv, object_count, annotated


def run_self_test(args: argparse.Namespace) -> int:
    """Menjalankan uji pipeline tanpa webcam dan tanpa jendela GUI."""
    frame = generate_self_test_image(args.width, args.height)
    dashboard, mask, _hsv, object_count, _annotated = process_frame(
        frame=frame,
        hsv_ranges=COLOR_PRESETS["Merah"],
        target_name="Merah",
        min_area=args.min_area,
        kernel_size=args.kernel_size,
        fps=0.0,
        message="Self-test selesai",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "self_test_result.jpg"
    mask_path = output_dir / "self_test_mask.png"

    ok_result = cv2.imwrite(str(result_path), dashboard)
    ok_mask = cv2.imwrite(str(mask_path), mask)
    if not ok_result or not ok_mask:
        print("ERROR: gagal menyimpan hasil self-test.", file=sys.stderr)
        return 1

    # Harus mendeteksi dua objek utama. Noise kecil diabaikan oleh min_area.
    if object_count != 2:
        print(
            f"ERROR SELF-TEST: seharusnya 2 objek, tetapi terdeteksi {object_count}.",
            file=sys.stderr,
        )
        return 1

    print("SELF-TEST BERHASIL")
    print(f"Jumlah objek terdeteksi: {object_count}")
    print(f"Hasil dashboard: {result_path.resolve()}")
    print(f"Hasil mask: {mask_path.resolve()}")
    return 0


def run_application(args: argparse.Namespace) -> int:
    """Menjalankan aplikasi deteksi warna secara real-time."""
    source = parse_source(args.source)
    capture = create_video_source(
        source=source,
        width=args.width,
        height=args.height,
        demo=args.demo,
    )

    if not capture.isOpened():
        print("ERROR: webcam atau video tidak dapat dibuka.", file=sys.stderr)
        print("Solusi yang dapat dicoba:", file=sys.stderr)
        print("1. Tutup aplikasi lain yang sedang memakai kamera.", file=sys.stderr)
        print("2. Izinkan akses kamera untuk Python/Terminal di Windows Settings.", file=sys.stderr)
        print("3. Coba kamera lain: python deteksi_warna_realtime.py --source 1", file=sys.stderr)
        print("4. Uji tanpa kamera: python deteksi_warna_realtime.py --demo", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    target_name = args.color
    hsv_ranges = COLOR_PRESETS[target_name]
    clicked_point: Optional[tuple[int, int]] = None
    status_message = "Arahkan objek ke kamera. Tekan 1-6 atau klik warna target."
    message_expiry = time.perf_counter() + 5.0
    previous_time = time.perf_counter()
    smoothed_fps = 0.0

    def mouse_callback(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        nonlocal clicked_point
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_point = (x, y)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    try:
        while True:
            success, frame = capture.read()
            if not success or frame is None or frame.size == 0:
                print("INFO: frame tidak dapat dibaca atau video telah selesai.")
                break

            frame = resize_letterbox(frame, args.width, args.height)
            frame_height, frame_width = frame.shape[:2]

            # Hitung FPS yang dihaluskan agar angka tidak terlalu berubah-ubah.
            current_time = time.perf_counter()
            elapsed = max(current_time - previous_time, 1e-9)
            instantaneous_fps = 1.0 / elapsed
            previous_time = current_time
            smoothed_fps = (
                instantaneous_fps
                if smoothed_fps == 0.0
                else 0.90 * smoothed_fps + 0.10 * instantaneous_fps
            )

            # HSV awal untuk membaca warna pada titik hasil klik.
            blurred_for_sampling = cv2.GaussianBlur(frame, (7, 7), 0)
            hsv_for_sampling = cv2.cvtColor(blurred_for_sampling, cv2.COLOR_BGR2HSV)

            if clicked_point is not None:
                click_x, click_y = clicked_point
                clicked_point = None

                # Klik hanya valid pada area frame utama. Dashboard preview berada di bawah frame.
                if 0 <= click_x < frame_width and 0 <= click_y < frame_height:
                    radius = 3
                    x1 = max(0, click_x - radius)
                    x2 = min(frame_width, click_x + radius + 1)
                    y1 = max(0, click_y - radius)
                    y2 = min(frame_height, click_y + radius + 1)
                    sample = hsv_for_sampling[y1:y2, x1:x2]
                    median_hsv = np.median(sample.reshape(-1, 3), axis=0).astype(int)
                    hue, saturation, value = map(int, median_hsv)

                    if saturation < 45 or value < 45:
                        status_message = (
                            "Kalibrasi ditolak: klik warna yang lebih cerah dan lebih pekat."
                        )
                    else:
                        hsv_ranges = custom_ranges_from_hsv(hue, saturation, value)
                        target_name = f"Kustom HSV({hue},{saturation},{value})"
                        status_message = (
                            f"Kalibrasi berhasil pada HSV({hue},{saturation},{value})."
                        )
                    message_expiry = time.perf_counter() + 4.0

            visible_message = (
                status_message if time.perf_counter() <= message_expiry else ""
            )

            dashboard, _mask, _hsv, _object_count, _annotated = process_frame(
                frame=frame,
                hsv_ranges=hsv_ranges,
                target_name=target_name,
                min_area=args.min_area,
                kernel_size=args.kernel_size,
                fps=smoothed_fps,
                message=visible_message,
            )

            cv2.imshow(WINDOW_NAME, dashboard)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break

            if key in KEY_TO_COLOR:
                target_name = KEY_TO_COLOR[key]
                hsv_ranges = COLOR_PRESETS[target_name]
                status_message = f"Target warna diubah menjadi {target_name}."
                message_expiry = time.perf_counter() + 3.0

            if key in (ord("s"), ord("S")):
                try:
                    saved_path = save_screenshot(dashboard, output_dir)
                    status_message = f"Screenshot tersimpan: {saved_path.name}"
                    print(f"Screenshot tersimpan: {saved_path.resolve()}")
                except OSError as error:
                    status_message = str(error)
                    print(f"ERROR: {error}", file=sys.stderr)
                message_expiry = time.perf_counter() + 4.0

    except KeyboardInterrupt:
        print("\nProgram dihentikan melalui keyboard.")
    finally:
        capture.release()
        cv2.destroyAllWindows()

    return 0


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deteksi warna objek real-time menggunakan OpenCV dan HSV."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Indeks kamera atau path video. Default: 0",
    )
    parser.add_argument(
        "--color",
        choices=list(COLOR_PRESETS.keys()),
        default="Merah",
        help="Warna target awal. Default: Merah",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=960,
        help="Lebar frame pemrosesan. Default: 960",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=540,
        help="Tinggi permintaan resolusi webcam/demo. Default: 540",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=800.0,
        help="Luas contour minimum dalam piksel. Default: 800",
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=5,
        help="Ukuran kernel morphology. Angka ganjil disarankan. Default: 5",
    )
    parser.add_argument(
        "--output-dir",
        default=str(SCRIPT_DIR / "output"),
        help="Folder penyimpanan screenshot. Default: folder output di dalam project",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Gunakan video sintetis bergerak tanpa webcam.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Uji pipeline tanpa webcam dan tanpa membuka GUI.",
    )
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    if args.width < 320:
        raise ValueError("--width minimal 320 piksel.")
    if args.height < 240:
        raise ValueError("--height minimal 240 piksel.")
    if args.min_area <= 0:
        raise ValueError("--min-area harus lebih besar dari 0.")
    if args.kernel_size <= 0:
        raise ValueError("--kernel-size harus lebih besar dari 0.")


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    try:
        validate_arguments(args)
    except ValueError as error:
        parser.error(str(error))

    if args.self_test:
        return run_self_test(args)
    return run_application(args)


if __name__ == "__main__":
    raise SystemExit(main())
