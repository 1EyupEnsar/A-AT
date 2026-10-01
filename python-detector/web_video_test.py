"""
Web panelindeki "Video Testi" ekranindan (Admin) tetiklenen, tek seferlik
video test scripti.

main.py'den farki: backend'e API uzerinden KAYIT GONDERMEZ (cooldown, konum,
siddet, tekrar-onleme yok) - amaci sadece 3 uzman modelin bir videoyu nasil
gordugunu HIZLICA gorsellestirmektir. Cikti, tarayicida dogrudan oynatilabilen
H.264/mp4 formatinda uretilir (ffmpeg'e ham kare akisi olarak pipe edilir).

Kullanim:
    python web_video_test.py <girdi_video_yolu> <cikti_mp4_yolu>

Sonuc, tek satirlik bir JSON olarak stdout'a yazilir; backend (ASP.NET Core)
bu JSON'u okuyup ekranda ozet olarak gosterir.
"""
import json
import os
import subprocess
import sys

import cv2
import imageio_ffmpeg
from ultralytics import YOLO

import config

RENKLER = {
    "Cukur": (0, 0, 255),          # kirmizi (BGR)
    "HasarliTabela": (255, 0, 0),  # mavi
    "TasanCopKutusu": (0, 255, 0), # yesil
}


def gecerli_mi(ariza_turu, x1, y1, x2, y2):
    """main.py'deki ile AYNI genislik/yukseklik filtresi - gercek sistemin
    ne gosterip ne gostermeyecegini birebir yansitmasi icin."""
    if ariza_turu not in config.EN_BOY_FILTRESI_UYGULANACAK_TURLER:
        return True
    genislik, yukseklik = x2 - x1, y2 - y1
    if yukseklik <= 0:
        return False
    return (genislik / yukseklik) >= config.MIN_EN_BOY_ORANI


def main():
    if len(sys.argv) < 3:
        print(json.dumps({"hata": "kullanim: web_video_test.py <girdi> <cikti_mp4>"}))
        sys.exit(1)

    girdi_yolu, cikti_yolu = sys.argv[1], sys.argv[2]

    cap = cv2.VideoCapture(girdi_yolu)
    if not cap.isOpened():
        print(json.dumps({"hata": f"Video acilamadi: {girdi_yolu}"}))
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(cikti_yolu), exist_ok=True)

    modeller = [(turu, YOLO(yol)) for turu, yol in config.MODEL_PATHS.items()]

    ffmpeg_cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "-",
        "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        cikti_yolu,
    ]
    proc = subprocess.Popen(
        ffmpeg_cmd, stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    sayaclar = {turu: 0 for turu in config.MODEL_PATHS}
    kare_no = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            for ariza_turu, model in modeller:
                sonuclar = model.predict(frame, conf=config.CONFIDENCE_THRESHOLD, verbose=False)[0]
                for kutu in sonuclar.boxes:
                    x1, y1, x2, y2 = [int(v) for v in kutu.xyxy[0].tolist()]
                    if not gecerli_mi(ariza_turu, x1, y1, x2, y2):
                        continue
                    guven = float(kutu.conf[0])
                    renk = RENKLER[ariza_turu]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), renk, 2)
                    cv2.putText(
                        frame, f"{ariza_turu} {guven:.2f}", (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, renk, 2,
                    )
                    sayaclar[ariza_turu] += 1
            proc.stdin.write(frame.tobytes())
            kare_no += 1
    except BrokenPipeError:
        print(json.dumps({"hata": "ffmpeg beklenmedik sekilde sonlandi (kodek sorunu olabilir)"}))
        sys.exit(1)
    finally:
        cap.release()
        try:
            proc.stdin.close()
        except OSError:
            pass
        proc.wait()

    if proc.returncode != 0 or not os.path.exists(cikti_yolu):
        print(json.dumps({"hata": "ffmpeg video uretemedi"}))
        sys.exit(1)

    print(json.dumps({"kare_sayisi": kare_no, "tespitler": sayaclar}))


if __name__ == "__main__":
    main()
