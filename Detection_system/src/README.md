src/ — Asosiy Python skriptlari

Fayllar:

- SMZ_Complete_Detection.py
  - Maqsadi: oddiy YOLOv8 asosidagi real-time defekt aniqlash skripti.
  - Vazifalari: kamera oqimini o‘qish, model orqali aniqlash, ekran chiqarish, natijalarni CSV ga saqlash va grafik ko‘rsatish.
  - Asosiy funksiyalar: DefectDetectionSystem class, process_video(), save_results(), visualize_statistics(), generate_report().
  - Ishga tushirish: `python SMZ_Complete_Detection.py`
  - Kutubxonalar: ultralytics, opencv-python, numpy, pandas, matplotlib, seaborn, scipy

- SMZ_Desktop_Pro.py
  - Maqsadi: professionalroq demo uchun kengaytirilgan skript (rasmlar saqlash, SQLite DB, PDF/Excel export, sessiyalar).
  - Vazifalari: deteksiyalarni fayl tizimiga saqlash (images/, videos/, reports/), bazaga yozish, PDF hisobot yaratish.
  - Asosiy klasslar: DatabaseManager, DefectDetectionSystemPro.
  - Ishga tushirish: `python SMZ_Desktop_Pro.py`
  - Qo‘shimcha kutubxonalar: openpyxl, pillow (PIL), matplotlib.backends.backend_pdf

Tavsiya:
- Agar loyihani birlashtirish kerak bo‘lsa, SMZ_Desktop_Pro.py ichidagi DB va fayl saqlash logikasini SMZ_Complete_Detection.py ga integratsiya qilish mumkin.
- Model fayli (`yolov8n.pt`) models/ ichida bo‘lishi kerak va kernel/IDE bilan bir xil Python muhitida ultralytics o‘rnatilganligiga eʼtibor bering.