"""
Bayrampasa videosunda Cukur (segmentasyon) modelinin yanlislikla "cukur"
dedigi TUM kareler incelendi: 16 tespitin TAMAMI 4 fiziksel konumdaki logar/
rogar kapagiydi, sifir gercek cukur (bkz. rapor - dairesellik/geometrik filtre
denendi ve ISE YARAMADIGI olculdu, config.py'deki not). Cozum veri tabanli:
bu 4 konumun her birinden birkac kare, ETIKETSIZ NEGATIF olarak cukur veri
setine ekleniyor (hard_negatif_cikar.py'nin cop modelinde kullanilan yontemiyle
ayni mantik, ama zaman araligi yerine BILINEN KONUM kareleri ile).
"""
import os

import cv2

import config

VIDEO = r"C:\Users\lasto\OneDrive\Desktop\bayrampasa1.mp4"
# Her biri farkli fiziksel logar/rogar kapaginin gorundugu kare kumeleri
# (Bayrampasa videosunda best_cukur_seg.pt yanlis pozitif verdigi kareler).
KONUM_KARELERI = [906, 909, 912, 915, 918, 2355, 2358, 2361, 2364, 2367,
                  2481, 2484, 3873, 3975, 3984]
CIKTI = os.path.join(config.BASE_DIR, "dataset", "logar_negatifleri")


def main():
    for s in ("train", "valid"):
        os.makedirs(os.path.join(CIKTI, s, "images"), exist_ok=True)
        os.makedirs(os.path.join(CIKTI, s, "labels"), exist_ok=True)

    hedef = set(KONUM_KARELERI)
    cap = cv2.VideoCapture(VIDEO)
    n = yazilan = 0
    while True:
        ok, kare = cap.read()
        if not ok:
            break
        n += 1
        if n not in hedef:
            continue
        split = "valid" if n % 5 == 0 else "train"
        stem = f"logarneg_{n:05d}"
        cv2.imwrite(os.path.join(CIKTI, split, "images", stem + ".jpg"), kare)
        open(os.path.join(CIKTI, split, "labels", stem + ".txt"), "w").close()
        cv2.imwrite(os.path.join(CIKTI, split, "images", stem + "_flip.jpg"), cv2.flip(kare, 1))
        open(os.path.join(CIKTI, split, "labels", stem + "_flip.txt"), "w").close()
        yazilan += 2
    cap.release()
    print(f"{yazilan} logar negatif karesi yazildi -> {CIKTI}")


if __name__ == "__main__":
    main()
