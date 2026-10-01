"""bagcilar_test.py'nin tek-model versiyonu - her model ayri calisir, kesinti
riskini azaltir. Kullanim: python bagcilar_tek_model.py <model_yolu> <cikti.json>"""
import json
import sys

import cv2
from ultralytics import YOLO

VIDEO = r"C:\Users\lasto\AkilliSehirArizaTespit\python-detector\videos\kaynak\BağcılarYol.mp4"
ADIM = 15
CONF = 0.5

model_yolu, cikti = sys.argv[1], sys.argv[2]
m = YOLO(model_yolu)
cap = cv2.VideoCapture(VIDEO)
n = 0
sonuc = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    n += 1
    if n % ADIM:
        continue
    r = m.predict(f, conf=CONF, verbose=False)[0]
    for b in r.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        sonuc.append([n, float(b.conf[0]), (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
cap.release()
json.dump({"model": model_yolu, "orneklenen_kare": n // ADIM, "tespit": sonuc}, open(cikti, "w"))
print(f"{model_yolu}: {len(sonuc)} tespit, {n // ADIM} kare tarandi")
