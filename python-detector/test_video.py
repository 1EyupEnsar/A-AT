"""
Herhangi bir videoyu 3 uzman modelden (Cukur/HasarliTabela/TasanCopKutusu)
gecirip kutucuklu (bounding box) bir cikti videosu ureten, tek basina
calistirilabilir arac.

KULLANIM (bu klasorde bir terminal/PowerShell acip):
    .venv\\Scripts\\python.exe test_video.py videos\\benim_videom.mp4

Video dosyasi belirtilmezse config.py'daki VIDEO_PATH kullanilir:
    .venv\\Scripts\\python.exe test_video.py

Cikti, ayni klasorde "<video_adi>_test_sonucu.avi" olarak kaydedilir ve
otomatik olarak varsayilan video oynaticida acilir.
"""
import os
import sys

import cv2
from ultralytics import YOLO

import config

RENKLER = {
    "Cukur": (0, 0, 255),          # kirmizi (BGR)
    "HasarliTabela": (255, 0, 0),  # mavi
    "TasanCopKutusu": (0, 255, 0), # yesil
}


def main():
    video_yolu = sys.argv[1] if len(sys.argv) > 1 else config.VIDEO_PATH
    if not os.path.exists(video_yolu):
        print(f"HATA: Video bulunamadi: {video_yolu}")
        return

    print("Modeller yukleniyor...")
    modeller = [(turu, YOLO(yol)) for turu, yol in config.MODEL_PATHS.items()]

    cap = cv2.VideoCapture(video_yolu)
    if not cap.isOpened():
        print(f"HATA: Video acilamadi: {video_yolu}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    taban_ad = os.path.splitext(os.path.basename(video_yolu))[0]
    cikti_yolu = f"{taban_ad}_test_sonucu.avi"
    writer = cv2.VideoWriter(cikti_yolu, cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h))

    print(f"Isleniyor: {video_yolu}")
    kare_no = 0
    while True:
        basarili, frame = cap.read()
        if not basarili:
            break
        for ariza_turu, model in modeller:
            sonuclar = model.predict(frame, conf=config.CONFIDENCE_THRESHOLD, verbose=False)[0]
            for kutu in sonuclar.boxes:
                x1, y1, x2, y2 = [int(v) for v in kutu.xyxy[0].tolist()]
                guven = float(kutu.conf[0])
                renk = RENKLER[ariza_turu]
                cv2.rectangle(frame, (x1, y1), (x2, y2), renk, 2)
                cv2.putText(
                    frame, f"{ariza_turu} {guven:.2f}", (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, renk, 2,
                )
        writer.write(frame)
        kare_no += 1

    cap.release()
    writer.release()
    print(f"{kare_no} kare islendi -> {cikti_yolu}")
    print("Video aciliyor...")
    os.startfile(cikti_yolu)


if __name__ == "__main__":
    main()
