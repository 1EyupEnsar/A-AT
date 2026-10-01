"""BagcilarYol.mp4'teki KasisBozuk yanlis pozitiflerini (parke tasi, islak
zemin, lastik izi, asfalt yama - bkz. rapor Bolum 12.9.2) zor-negatif olarak
cikarir. logar_negatif_cikar.py ile AYNI yontem: her olayin temsilci karesi +
yatay aynali kopyasi, BOS etiket dosyasiyla dataset/kasis_negatifleri/{train,valid}
klasorune yazilir. %85/%15 train/valid ayrimi."""
import json
import os

import cv2

VIDEO = r"C:\Users\lasto\AkilliSehirArizaTespit\python-detector\videos\kaynak\BağcılarYol.mp4"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "kasis_negatifleri")


def kumele(liste, kare_bosluk=30):
    liste = sorted(liste, key=lambda x: x["kare"])
    kumeler = []
    for t in liste:
        if kumeler and t["kare"] - kumeler[-1][-1]["kare"] <= kare_bosluk:
            kumeler[-1].append(t)
        else:
            kumeler.append([t])
    return kumeler


def main():
    veri = json.load(open("bagcilar_kasis_detay.json"))
    kasis = [t for t in veri["tespit"] if t["tur"] == "KasisBozuk"]
    kumeler = kumele(kasis)
    temsilciler = [max(k, key=lambda x: x["guven"])["kare"] for k in kumeler]
    print(f"{len(kasis)} tespit, {len(kumeler)} kume, {len(temsilciler)} temsilci kare")

    for split in ("train", "valid"):
        os.makedirs(os.path.join(OUT_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR, split, "labels"), exist_ok=True)

    cap = cv2.VideoCapture(VIDEO)
    yazilan = 0
    for i, kare_no in enumerate(temsilciler):
        split = "valid" if i % 7 == 0 else "train"  # ~%15 valid
        cap.set(cv2.CAP_PROP_POS_FRAMES, kare_no - 1)
        ok, frame = cap.read()
        if not ok:
            continue
        for ek, im in (("", frame), ("_ayna", cv2.flip(frame, 1))):
            stem = f"bgcl_kasisneg_{kare_no}{ek}"
            cv2.imwrite(os.path.join(OUT_DIR, split, "images", stem + ".jpg"), im)
            open(os.path.join(OUT_DIR, split, "labels", stem + ".txt"), "w").close()
            yazilan += 1
    cap.release()
    print(f"{yazilan} goruntu yazildi (negatif+ayna) -> {OUT_DIR}")


if __name__ == "__main__":
    main()
