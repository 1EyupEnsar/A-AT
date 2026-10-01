"""Kasis modelini BagcilarYol.mp4 uzerinde, main.py'deki GERCEK uretim
mantigiyla (MIN_EN_BOY_ORANLARI genislik/yukseklik filtresi + GecitBozuk
icin gecit_analiz.cizgi_gorunurlugu_skoru ile Saglam/Bozuk/Kullanilamaz
ayrimi) test eder. Her ornek karede, her tespit icin nihai durumu kaydeder."""
import json

import cv2
from ultralytics import YOLO

import config
import gecit_analiz

VIDEO = r"C:\Users\lasto\AkilliSehirArizaTespit\python-detector\videos\kaynak\BağcılarYol.mp4"
ADIM = 15
CONF = 0.5
SINIF_ADLARI = config.MODEL_SINIF_ADLARI["Kasis"]

model = YOLO("runs/detect/uzman_kasis/weights/best.pt")
cap = cv2.VideoCapture(VIDEO)
n = 0
sonuc = []
while True:
    ok, frame = cap.read()
    if not ok:
        break
    n += 1
    if n % ADIM:
        continue
    r = model.predict(frame, conf=CONF, verbose=False)[0]
    for b in r.boxes:
        sinif_id = int(b.cls[0])
        ham_tur = SINIF_ADLARI[sinif_id]
        guven = float(b.conf[0])
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        genislik, yukseklik = x2 - x1, y2 - y1

        esik = config.MIN_EN_BOY_ORANLARI.get(ham_tur)
        if esik is not None and (yukseklik <= 0 or (genislik / yukseklik) < esik):
            continue  # geometrik filtre eledi

        nihai_tur = ham_tur
        skor = None
        if ham_tur == "YayaGecidi":
            pay = 10
            h, w = frame.shape[:2]
            kx1, ky1 = max(0, int(x1) - pay), max(0, int(y1) - pay)
            kx2, ky2 = min(w, int(x2) + pay), min(h, int(y2) + pay)
            kirpik = frame[ky1:ky2, kx1:kx2]
            skor = gecit_analiz.cizgi_gorunurlugu_skoru(kirpik)
            if skor >= config.GECIT_SAGLAM_ESIK:
                continue  # saglam gecit, ariza degil, raporlanmaz
            elif skor < config.GECIT_KULLANILAMAZ_ESIK:
                nihai_tur = "GecitKullanilamaz"
            else:
                nihai_tur = "GecitBozuk"

        sonuc.append({
            "kare": n, "tur": nihai_tur, "guven": round(guven, 3),
            "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2,
            "w": genislik, "h": yukseklik,
            "skor": round(skor, 3) if skor is not None else None,
        })
cap.release()

json.dump({"orneklenen_kare": n // ADIM, "tespit": sonuc}, open("bagcilar_kasis_detay.json", "w"), indent=2)
print(f"toplam {len(sonuc)} tespit (filtre sonrasi), {n // ADIM} kare tarandi")
from collections import Counter
print(Counter(s["tur"] for s in sonuc))
