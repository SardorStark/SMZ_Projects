AI_for_Predection — Prediction pipeline

Ushbu papka rulon og'irligi/uzunligini va boshqa atributlarni bashorat qilish uchun zarur modul va skriptlarni o'z ichiga oladi.

Fayllar:
- data_fetcher.py   — BJ-SRI API dan ma'lumot olish uchun klient (token/endpoint sozlanadi)
- excel_loader.py   — Excel fayllarni yuklash va birlashtirish
- eda_anomaly.py    — EDA va anomaliya aniqlash (IQR + IsolationForest) va vizualizatsiyalar
- modeling.py       — feature engineering va LightGBM asosidagi baseline regressiya

Ishga tushirish tezkor tartibi:
1) BJ-SRI API sozlamalarini environment ga qo'ying: BJ_SRI_API_KEY va BJ_SRI_BASE_URL
2) Rawmadatalarni olish: python -c "from ai_for_predection.data_fetcher import BJSRIFetcher; BJSRIFetcher().fetch_all(...)"
3) Excellarni merge qilish: from ai_for_predection.excel_loader import load_excel_folder
4) EDA va anomaly: python ai_for_predection/eda_anomaly.py --input merged.csv
5) Model trening: python ai_for_predection/modeling.py --input cleaned.csv

Keyingi bosqichda siz bergan aniq formulalar va qo'shimcha ustunlar asosida featurelarni moslashtirib chiqamiz.
