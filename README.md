# SMZ Defect Detection — Project Overview

SMZ Defect Detection loyihasi ishlab chiqarish chizig'ida mahsulot yuzasidagi defektlarni avtomatik aniqlash, saqlash, tahlil va hisobotlar yaratish uchun mo'ljallangan.

---

## Tezkor ko'rinish (Quickstart)

1. Kutubxonalarni o'rnatish:

```bash
python -m pip install --upgrade pip
pip install ultralytics opencv-python pandas matplotlib seaborn openpyxl pillow
```

2. Modelni joylashtirish: `models/` ichiga `yolov8n.pt` yoki o'zingizning `best.pt` faylingizni qo'ying.

3. Test rasm qo'yish: `Detection_system/test_images/` papkaga `.jpg` yoki `.png` fayllarni yuklang.

4. Testni ishga tushirish:

```bash
cd C:\Users\Sardor\Desktop\SMZ_Projects\Detection_system
python src\process_images_test.py --input test_images --model models\yolov8n.pt --conf 0.25
# yoki professional pipeline uchun:
python src\advanced_pipeline.py --input test_images --models models\yolov8n.pt --conf 0.25 --tta --scales 640 800
```

Natijalar `Detection_system/detection_results/` ichida saqlanadi (images/defects, images/full_frames, data/detection_results.csv, database.db).

---

## Loyihaning maqsadi

- Real-time yoki batch rejimda mahsulot yuzasidagi defektlarni aniqlash.
- Har bir aniqlangan defekt uchun crop va thumbnail yaratish.
- Barcha meta-ma'lumotlarni CSV va SQLite bazasida saqlash.
- PDF va Excel hisobotlarini avtomatik generatsiya qilish.
- Demo desktop uchun GUI va eksport imkoniyatlarini taqdim etish.

---

## Papka tuzilmasi

```
SMZ_Projects/
├─ Detection_system/
│  ├─ models/                # Model fayllari (yolov8n.pt)
│  ├─ src/                   # Asosiy skriptlar (process_images_test.py, advanced_pipeline.py,...)
│  ├─ test_images/           # Siz yuklagan test rasmlar
│  ├─ detection_results/     # Output (rasmlar, CSV, DB, reports)
│  └─ notebooks/             # Jupyter notebooklar
└─ README.md
```

---

## Etakchi funksiyalar

- `process_images_test.py` — soddalashtirilgan test skript: bitta papkadan rasmlarni o'qiydi va aniqlangan defektlarni saqlaydi.
- `advanced_pipeline.py` — professional pipeline: TTA, multi-scale, ensemble, CLAHE preprocessing, weighted merge va SQLite/CSV eksport.
- `SMZ_Desktop_Pro.py` — demo desktop uchun kengaytirilgan funksiyalar (session, PDF, Excel).

---

## Label qilish va trening

1. Tasvirlarni YOLO formatda label qiling: `labels/train/*.txt` va `labels/val/*.txt` (har bir rasmga mos .txt, format: `class x_center y_center width height` — normalizatsiyalangan 0..1).
2. `data.yaml` namunasi:

```yaml
train: ../images/train
val:   ../images/val
nc: 1
names: ['yoriq']
```

3. Trening buyruqlari:

```bash
# CLI
yolo task=detect mode=train data=path/to/data.yaml model=models/yolov8n.pt epochs=50 imgsz=640 batch=16 project=runs/train name=exp1

# Python API
from ultralytics import YOLO
model = YOLO('models/yolov8n.pt')
model.train(data='path/to/data.yaml', epochs=50, imgsz=640, batch=16)
```

---

## Tavsiyalar va kelajak yondashuvlar

- Rasm fayllarini CSV ichiga base64 qilib saqlamang — faqat fayl yo'lini saqlang.
- Real-time uchun GPU (CUDA) ishlatish tavsiya etiladi.
- Keyingi qo'shimchalar: GUI (PyQt5), Active Learning loop, mask-segmentation, LightGBM post-classifier.

---

## Kontakt va Hujjat

Agar qo'shimcha o'zgartirish yoki packaging (.exe) kerak bo'lsa, xabar bering — men yordam beraman.

PRESENTATION: `PRESENTATION.md` — qisqacha slaydlar fayli loyihaning muhim qismlarini taqdim etadi.
