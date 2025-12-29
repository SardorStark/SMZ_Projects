data/ — Chiqish maʼlumotlari va misollar

Mavjud fayl:

- detection_results.csv — oldingi test/aniqlash natijalari CSV formatida saqlangan bo‘lishi mumkin. CSV ichida odatda quyidagi ustunlar bo‘ladi: timestamp, frame, class, confidence, x1, y1, x2, y2, area, defect_image_path

Tavsiyalar:
- Rasm fayllarini to‘g‘ridan-to‘g‘ri CSV ichiga kodlash (base64) qilishdan saqlaning — bu faylni juda katta va sekin qiladi.
- Rasm fayllarini `images/` papkasida saqlash va CSV yoki SQLite ichida faqat fayl yo‘lini saqlash eng to‘g‘ri amaliyot.
- Agar loyiha demo desktop bo‘lsa, SQLite yordamida rasm yo‘llari va meta-maʼlumotlarni saqlashni tavsiya qilamiz.