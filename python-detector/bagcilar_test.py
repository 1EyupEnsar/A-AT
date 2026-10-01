"""
BağcılarYol.mp4 (34 dk, 61867 kare) uzerinde eski/yeni model ciftlerini
karsilastirir. Her 15. kare orneklenir (~4124 kare/model). Her model icin
tespit sayisi + kare numaralari JSON'a yazilir (sonradan kume/gorsel analiz icin).
"""
import json
import sys

import cv2
from ultralytics import YOLO

VIDEO = r"C:\Users\lasto\AkilliSehirArizaTespit\python-detector\videos\kaynak\BağcılarYol.mp4"
ADIM = 15
CONF = 0.5


def tara(model_yolu):
    m = YOLO(model_yolu)
    cap = cv2.VideoCapture(VIDEO)
    n = 0
    sonuc = []  # (kare_no, guven, kutu_merkezi_x, kutu_merkezi_y, genislik, yukseklik)
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
    return n, sonuc


if __name__ == "__main__":
    etiket, yol1, yol2, cikti = sys.argv[1:5]
    toplam_kare1, s1 = tara(yol1)
    toplam_kare2, s2 = tara(yol2)
    json.dump({"etiket": etiket, "yol1": yol1, "yol2": yol2,
               "orneklenen_kare": toplam_kare1 // ADIM,
               "tespit1": s1, "tespit2": s2},
              open(cikti, "w"))
    print(f"{etiket}: model1({yol1})={len(s1)} tespit, model2({yol2})={len(s2)} tespit, "
          f"{toplam_kare1 // ADIM} kare tarandi")
