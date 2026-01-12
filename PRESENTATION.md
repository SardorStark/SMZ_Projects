# SMZ Project — Presentation Slides

---

## Slide 1 — Loyihaning maqsadi

- Mahsulot yuzasidagi defektlarni aniqlash va klassifikatsiya
- Real‑time monitoring va tarixiy tahlil
- Professional hisobot va eksport

---

## Slide 2 — Arxitektura (high-level)

- Modellar: YOLOv8 (yolov8n/ s / m)
- Pipeline: Preprocess (CLAHE) → Inference (TTA, multi‑scale) → Postprocess (merge, morph) → Save (crop, thumbnail)
- Storage: filesystem + SQLite + CSV/Excel/PDF

---

## Slide 3 — Demo oqimi

1. Rasm yoki video yuklash
2. Model orqali aniqlash
3. Aniqlangan defektlarni papkaga saqlash
4. CSV/DB va PDF hisobot yaratish

---

## Slide 4 — Trening va label qilish

- YOLO formatdagi label fayllarini yarating
- `data.yaml` ning tuzilishi va training buyruqlari
- Active learning va retraining strategiyasi

---

## Slide 5 — Keyingi qadamlar

- GUI (PyQt5) yaratish va packaging (.exe)
- Segmentatsiya va severity scoring
- Monitoring dashboard (Grafana yoki web UI)

---

## Slide 6 — Aloqa

- Loyihani kengaytirish va testlar uchun bog'laning.
- Rekvizit: `C:\Users\Sardor\Desktop\SMZ_Projects\Detection_system`
