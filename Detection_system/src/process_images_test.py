"""
process_images_test.py

Qulay test skripti: test_images/ papkasidagi barcha .jpg/.png rasmlarni model bilan qayta ishlaydi,
har bir aniqlangan defektni images/defects ga saqlaydi, to'liq annotatsiyalangan kadrni images/full_frames ga saqlaydi
va detection_results/data/detection_results.csv ga meta-ma'lumotlarni qo'shadi.

Ishlatish:
python process_images_test.py --input test_images --model ../models/yolov8n.pt --conf 0.25

"""

import argparse
from pathlib import Path
import time
from datetime import datetime
import os
import sys

try:
    from ultralytics import YOLO
except Exception as e:
    print("ERROR: ultralytics paketini o'rnating: pip install ultralytics")
    raise

import cv2
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / 'test_images'
OUTPUT_DIR = PROJECT_ROOT / 'detection_results'
DATA_DIR = OUTPUT_DIR / 'data'
IMAGES_DEFECTS = OUTPUT_DIR / 'images' / 'defects'
IMAGES_FULL = OUTPUT_DIR / 'images' / 'full_frames'
THUMBS = OUTPUT_DIR / 'images' / 'thumbnails'
CSV_PATH = DATA_DIR / 'detection_results.csv'

# Ensure directories
for p in (DATA_DIR, IMAGES_DEFECTS, IMAGES_FULL, THUMBS, OUTPUT_DIR / 'videos', OUTPUT_DIR / 'reports', OUTPUT_DIR / 'export'):
    p.mkdir(parents=True, exist_ok=True)


def save_detection_row(rows, row):
    rows.append(row)


def process_image(model, img_path, conf_thres=0.25, start_idx=0):
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"⚠️ Rasm o'qib bo'lmadi: {img_path}")
        return [], 0

    # Run inference with given confidence threshold
    results = model(img_bgr, conf=conf_thres, verbose=False)
    annotated = results[0].plot() if hasattr(results[0], 'plot') else img_bgr.copy()

    rows = []
    full_frame_rel = None

    # Save full annotated frame
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    frame_filename = f"{img_path.stem}_{timestamp}.jpg"
    full_frame_path = IMAGES_FULL / frame_filename
    cv2.imwrite(str(full_frame_path), annotated)
    full_frame_rel = str(full_frame_path.relative_to(OUTPUT_DIR))

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return rows, 0

    defect_id = start_idx
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        # safe extraction of coordinates (torch tensor or numpy)
        coords = box.xyxy[0]
        try:
            coords = coords.cpu().numpy()
        except Exception:
            pass
        x1, y1, x2, y2 = map(int, coords)
        area = (x2 - x1) * (y2 - y1)

        # Crop and save defect image
        padding = 10
        h, w = img_bgr.shape[:2]
        xa = max(0, x1 - padding)
        ya = max(0, y1 - padding)
        xb = min(w, x2 + padding)
        yb = min(h, y2 + padding)
        crop = img_bgr[ya:yb, xa:xb]

        defect_filename = f"{timestamp}_defect_{defect_id:05d}.jpg"
        defect_path = IMAGES_DEFECTS / defect_filename
        cv2.imwrite(str(defect_path), crop)

        # Thumbnail
        try:
            thumb = cv2.resize(crop, (150, 150))
            thumb_path = THUMBS / defect_filename
            cv2.imwrite(str(thumb_path), thumb)
        except Exception:
            thumb_path = ''

        row = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'source_image': str(img_path.relative_to(PROJECT_ROOT)),
            'frame_name': frame_filename,
            'frame_number': 0,
            'class_id': cls,
            'confidence': conf,
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'area': area,
            'defect_image_path': str(defect_path.relative_to(OUTPUT_DIR)),
            'full_frame_path': full_frame_rel
        }
        rows.append(row)
        defect_id += 1

    return rows, defect_id - start_idx


def append_rows_to_csv(rows, csv_path=CSV_PATH):
    if not rows:
        return
    df = pd.DataFrame(rows)
    if csv_path.exists():
        df.to_csv(csv_path, mode='a', header=False, index=False, encoding='utf-8-sig')
    else:
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"💾 {len(rows)} qator CSV ga qo'shildi: {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=str(DEFAULT_INPUT), help='Input folder with images')
    parser.add_argument('--model', type=str, default=str(PROJECT_ROOT / 'models' / 'yolov8n.pt'), help='YOLO model path')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"⚠️ Input papka topilmadi: {input_dir}. Iltimos rasm fayllarni shu papkaga joylang.")
        return

    print(f"🔍 Model yuklanmoqda: {args.model}")
    model = YOLO(str(args.model))

    image_files = sorted([p for p in input_dir.glob('*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    if not image_files:
        print(f"⚠️ {input_dir} ichida rasm topilmadi. .jpg yoki .png fayllarni joylang.")
        return

    total_detections = 0
    next_defect_id = 0
    all_rows = []

    for img_path in image_files:
        print(f"Processing: {img_path.name}")
        rows, count = process_image(model, img_path, conf_thres=args.conf, start_idx=next_defect_id)
        append_rows_to_csv(rows)
        total_detections += count
        next_defect_id += count

    print("\n✅ Yakun: ")
    print(f"   Rasm fayllar soni: {len(image_files)}")
    print(f"   Topilgan defektlar: {total_detections}")
    print(f"   Natijalar papkasi: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()