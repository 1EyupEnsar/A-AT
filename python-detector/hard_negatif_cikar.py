"""
Gercek sokak videosundan 'zor negatif' kare cikarir: hedef sinifin (orn.
tasan cop kutusu) OLMADIGI ama modelin yanlislikla tespit ettigi sahneler.
Kareler etiketsiz (bos label) olarak egitime eklenir (bkz. merge_dataset_separate.py).

Zaman bazli ayirma: videonun ILK kismi (--bolme_kare oncesi) egitime gider,
kalan kismi hic egitime girmez ve yalnizca adil test icin ayrilir.

Kullanim:
  python hard_negatif_cikar.py <video> <onek> --bolme_kare 2500 [--adim 10]
"""
import argparse
import os

import cv2

import config

CIKTI = os.path.join(config.BASE_DIR, "dataset", "hard_negatives")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("onek")
    ap.add_argument("--bolme_kare", type=int, required=True)
    ap.add_argument("--adim", type=int, default=10)
    ap.add_argument("--val_adim", type=int, default=5, help="her N. egitim karesi valid'e gider")
    a = ap.parse_args()

    for s in ("train", "valid"):
        os.makedirs(os.path.join(CIKTI, s, "images"), exist_ok=True)

    cap = cv2.VideoCapture(a.video)
    no, yazilan = 0, 0
    while True:
        ok, kare = cap.read()
        if not ok or no >= a.bolme_kare:
            break
        no += 1
        if no % a.adim:
            continue
        split = "valid" if (no // a.adim) % a.val_adim == 0 else "train"
        yol = os.path.join(CIKTI, split, "images", f"{a.onek}_{no:05d}")
        cv2.imwrite(yol + ".jpg", kare)
        # Yatay ayna: modelin kadraj kenarina (ozellikle sag) olan yanliligini kirar.
        cv2.imwrite(yol + "_flip.jpg", cv2.flip(kare, 1))
        yazilan += 2
    cap.release()
    print(f"{yazilan} zor negatif kare yazildi -> {CIKTI}")


if __name__ == "__main__":
    main()
