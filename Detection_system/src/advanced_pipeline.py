"""
advanced_pipeline.py

Professional advanced detection pipeline with:
- Multi-model ensemble support
- Test-time augmentation (flips, scales)
- Multi-scale inference
- Simple weighted box merging (ensemble)
- Preprocessing (CLAHE) and postprocessing (morphological refinement)
- Outputs: annotated frames, defect crops, thumbnails, CSV, SQLite

Usage:
python advanced_pipeline.py --input test_images --models models\yolov8n.pt --conf 0.25 --tta --scales 640 800

"""

import argparse
from pathlib import Path
import numpy as np
import cv2
import sqlite3
import pandas as pd
import time
from datetime import datetime

# Optional imports
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


# Utility functions
def iou(boxA, boxB):
    # boxes are [x1,y1,x2,y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(0, boxA[2]-boxA[0]) * max(0, boxA[3]-boxA[1])
    boxBArea = max(0, boxB[2]-boxB[0]) * max(0, boxB[3]-boxB[1])
    denom = float(boxAArea + boxBArea - interArea)
    return interArea / denom if denom > 0 else 0.0


def nms(boxes, scores, iou_threshold=0.45):
    # boxes: Nx4, scores: N
    idxs = np.argsort(scores)[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        rest = idxs[1:]
        ious = np.array([iou(boxes[i], boxes[j]) for j in rest])
        idxs = rest[ious <= iou_threshold]
    return keep


def weighted_merge(group):
    # group: list of (box, score, cls)
    # returns single merged box [x1,y1,x2,y2], avg_score, cls (majority)
    boxes = np.array([g[0] for g in group], dtype=float)
    scores = np.array([g[1] for g in group], dtype=float)
    classes = [g[2] for g in group]
    weights = scores / (scores.sum() + 1e-8)
    merged = (boxes * weights[:, None]).sum(axis=0)
    avg_score = float(scores.mean())
    # choose class by weighted vote
    cls = max(set(classes), key=classes.count)
    return merged.tolist(), avg_score, cls


class AdvancedDetector:
    def __init__(self, model_paths, conf=0.25, iou_thres=0.45, device=None):
        if YOLO is None:
            raise RuntimeError('ultralytics YOLO not available. Install ultralytics package')
        self.models = [YOLO(p) for p in model_paths]
        self.conf = conf
        self.iou_thres = iou_thres

    def predict_single(self, img, model, imgsz=None, conf=None):
        # model inference
        _conf = conf if conf is not None else self.conf
        # let ultralytics handle resizing via imgsz parameter
        if imgsz is not None:
            res = model(img, imgsz=imgsz, conf=_conf, verbose=False)
        else:
            res = model(img, conf=_conf, verbose=False)
        dets = []
        if len(res) > 0 and hasattr(res[0], 'boxes'):
            for box in res[0].boxes:
                coords = box.xyxy[0]
                try:
                    # handle tensor or numpy
                    coords = coords.cpu().numpy()
                except Exception:
                    pass
                x1, y1, x2, y2 = map(int, coords)
                cls = int(box.cls[0])
                confs = float(box.conf[0])
                dets.append(([x1, y1, x2, y2], confs, cls))
        return dets

    def tta_transforms(self, img):
        # generate augmented images (horizontal flip only)
        return [cv2.flip(img, 1)]

    def aggregate_detections(self, all_dets, merge_iou=0.5):
        # all_dets: list of det lists from models and augmentations
        flat = [d for dets in all_dets for d in dets]
        if not flat:
            return []
        boxes = np.array([f[0] for f in flat], dtype=float)
        scores = np.array([f[1] for f in flat], dtype=float)
        classes = np.array([f[2] for f in flat], dtype=int)
        # simple class-wise grouping then merging by IoU
        merged = []
        used = set()
        for i in range(len(flat)):
            if i in used:
                continue
            group = [flat[i]]
            used.add(i)
            for j in range(i+1, len(flat)):
                if j in used:
                    continue
                if classes[j] != classes[i]:
                    continue
                if iou(boxes[i], boxes[j]) >= merge_iou:
                    group.append(flat[j])
                    used.add(j)
            merged_box, avg_score, cls = weighted_merge(group)
            merged.append((list(map(int, merged_box)), float(avg_score), int(cls)))
        # final NMS to remove overlaps
        if not merged:
            return []
        mboxes = np.array([m[0] for m in merged], dtype=float)
        mscores = np.array([m[1] for m in merged], dtype=float)
        keep_idx = nms(mboxes, mscores, iou_threshold=self.iou_thres)
        final = [merged[i] for i in keep_idx]
        return final

    def predict(self, img, tta=False, scales=None):
        # scales: list of ints for multi-scale inference
        all_dets = []
        # per-model and per-scale predictions
        for m in self.models:
            if scales:
                for s in scales:
                    dets = self.predict_single(img, m, imgsz=s)
                    all_dets.append(dets)
            else:
                dets = self.predict_single(img, m)
                all_dets.append(dets)
            if tta:
                for aug in self.tta_transforms(img):
                    dets = self.predict_single(aug, m)
                    # if flipped, flip boxes back
                    if aug is not img:
                        h = img.shape[0]
                        dets_fixed = []
                        for b, sc, c in dets:
                            x1,y1,x2,y2 = b
                            nx1 = img.shape[1] - x2
                            nx2 = img.shape[1] - x1
                            dets_fixed.append(([nx1,y1,nx2,y2], sc, c))
                        all_dets.append(dets_fixed)
                    else:
                        all_dets.append(dets)
        # aggregate
        final = self.aggregate_detections(all_dets)
        return final


class Pipeline:
    def __init__(self, model_paths, output_dir, conf=0.25, iou=0.45, tta=False, scales=None):
        self.detector = AdvancedDetector(model_paths, conf=conf, iou_thres=iou)
        self.output_dir = Path(output_dir)
        self.conf = conf
        self.scales = scales
        self.tta = tta
        self.setup_dirs()
        self.init_db()

    def setup_dirs(self):
        self.images_def = self.output_dir / 'images' / 'defects'
        self.images_full = self.output_dir / 'images' / 'full_frames'
        self.thumbs = self.output_dir / 'images' / 'thumbnails'
        self.data_dir = self.output_dir / 'data'
        for d in (self.images_def, self.images_full, self.thumbs, self.data_dir, self.output_dir / 'reports', self.output_dir / 'export'):
            d.mkdir(parents=True, exist_ok=True)

    def init_db(self):
        self.db_path = self.output_dir / 'database.db'
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            source TEXT,
            class_id INTEGER,
            confidence REAL,
            x1 INTEGER,y1 INTEGER,x2 INTEGER,y2 INTEGER,
            area INTEGER,
            defect_image TEXT,
            full_frame TEXT
        )''')
        conn.commit()
        conn.close()

    def preprocess(self, img):
        # CLAHE on V channel
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l,a,b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return final

    def postprocess_crop(self, crop):
        # simple morphological denoise
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _,th = cv2.threshold(gray,0,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        clean = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
        # return original crop and mask (mask may be small)
        return crop, clean

    def annotate(self, img, detections):
        out = img.copy()
        for b, sc, cls in detections:
            x1,y1,x2,y2 = b
            color = (0,0,255)
            cv2.rectangle(out,(x1,y1),(x2,y2),color,2)
            label = f"{cls}:{sc:.2f}"
            cv2.putText(out,label,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
        return out

    def save_results(self, src_image_path, annotated, detections):
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        frame_name = f"{Path(src_image_path).stem}_{ts}.jpg"
        full_path = self.images_full / frame_name
        cv2.imwrite(str(full_path), annotated)
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()
        rows = []
        for idx, (b, sc, cls) in enumerate(detections):
            x1,y1,x2,y2 = b
            area = max(0,(x2-x1)) * max(0,(y2-y1))
            # save crop (clamp coords)
            h, w = annotated.shape[:2]
            x1c = max(0, min(w-1, x1))
            x2c = max(0, min(w, x2))
            y1c = max(0, min(h-1, y1))
            y2c = max(0, min(h, y2))
            if x2c <= x1c or y2c <= y1c:
                # invalid box, skip
                continue
            crop = annotated[y1c:y2c, x1c:x2c]
            crop_name = f"{ts}_def_{idx:04d}.jpg"
            crop_path = self.images_def / crop_name
            cv2.imwrite(str(crop_path), crop)
            # thumbnail
            try:
                th = cv2.resize(crop, (150,150))
                cv2.imwrite(str(self.thumbs / crop_name), th)
            except Exception:
                pass
            # DB insert
            c.execute('''INSERT INTO detections (timestamp, source, class_id, confidence, x1,y1,x2,y2,area,defect_image,full_frame)
                         VALUES (?,?,?,?,?,?,?,?,?,?,?)''', (
                         datetime.now().strftime('%Y-%m-%d %H:%M:%S'), str(src_image_path), int(cls), float(sc), int(x1),int(y1),int(x2),int(y2),int(area), str(crop_path.relative_to(self.output_dir)), str(full_path.relative_to(self.output_dir))
            ))
            rows.append({'timestamp':datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'source':str(src_image_path),'class_id':int(cls),'confidence':float(sc),'x1':x1,'y1':y1,'x2':x2,'y2':y2,'area':area,'defect_image':str(crop_path.relative_to(self.output_dir)),'full_frame':str(full_path.relative_to(self.output_dir))})
        conn.commit()
        conn.close()
        # append CSV
        csv_path = self.data_dir / 'detection_results.csv'
        df = pd.DataFrame(rows)
        if csv_path.exists():
            df.to_csv(csv_path, mode='a', header=False, index=False, encoding='utf-8-sig')
        else:
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    def run_on_image(self, img_path):
        img = cv2.imread(str(img_path))
        if img is None:
            print('Cannot read', img_path)
            return
        proc = self.preprocess(img)
        detections = self.detector.predict(proc, tta=self.tta, scales=self.scales)
        annotated = self.annotate(img, detections)
        self.save_results(img_path, annotated, detections)
        return len(detections)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='test_images')
    parser.add_argument('--models', type=str, required=True, help='Comma separated model paths')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--iou', type=float, default=0.45)
    parser.add_argument('--tta', action='store_true')
    parser.add_argument('--scales', nargs='*', type=int, default=None)
    parser.add_argument('--output', type=str, default='detection_results')
    args = parser.parse_args()

    model_paths = [m.strip() for m in args.models.split(',')]
    if YOLO is None:
        print('Install ultralytics: pip install ultralytics')
        return

    pipeline = Pipeline(model_paths, args.output, conf=args.conf, iou=args.iou, tta=args.tta, scales=args.scales)
    input_dir = Path(args.input)
    files = sorted([p for p in input_dir.glob('*') if p.suffix.lower() in ['.jpg','.jpeg','.png']])
    total = 0
    detections_total = 0
    for f in files:
        print('Processing', f.name)
        cnt = pipeline.run_on_image(f)
        detections_total += cnt if cnt else 0
        total += 1
    print(f"Done. Images: {total}. Detections: {detections_total}. Results: {pipeline.output_dir}")

if __name__ == '__main__':
    main()
