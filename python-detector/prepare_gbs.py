"""
GBS (Garbage Bin Status, Zenodo 14711706, CC BY 4.0; COCO: 0=overflow,
1=garbage_bin, 2=garbage) -> dataset/gbs_universe/{train,valid} (YOLO kutu).

Bilincli filtreleme (bkz. rapor): GBS'nin buyuk kismi ORIJINAL DEGIL -
  * "#.png"                    : Stable Diffusion sentetik goruntuler -> ALINMADI
  * b_/pad_/noise_/f_ on ekli  : ayni orijinallerin bulanik/dolgulu/gurultulu/
                                  aynali kopyalari -> ALINMADI (yeni bilgi
                                  vermez, train/valid sizintisi yaratir)
Sadece gercek orijinal .jpg'ler alinir:
  * en az bir 'overflow' kutusu olanlar -> pozitif (sadece overflow kutulari, sinif 0)
  * 'garbage_bin' olup 'overflow' OLMAYANLAR -> etiketsiz negatif (normal, dolu
    olmayan cop kutusu TasanCopKutusu sayilmamali)
Buyuk goruntuler (4080x3060 gibi) en uzun kenar 1280'e kucultulur (kutular
normalize oldugu icin degismez).
"""
import json
import os
import random
import re
import shutil
import zipfile

import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
ZIP = os.path.join(BASE, "dataset", "gbs_raw", "GBS.zip")
HEDEF = os.path.join(BASE, "dataset", "gbs_universe")
KOPYA_ONEKLERI = ("b_", "pad_", "noise_", "f_")
MAX_KENAR = 1280


def yaz(z, ad, hedef_stem, split):
    arr = np.frombuffer(z.read("Images/" + ad), np.uint8)
    im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if im is None:
        return False
    h, w = im.shape[:2]
    if max(h, w) > MAX_KENAR:
        k = MAX_KENAR / max(h, w)
        im = cv2.resize(im, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(HEDEF, split, "images", hedef_stem + ".jpg"), im)
    return True


def main():
    if os.path.exists(HEDEF):
        shutil.rmtree(HEDEF)
    for s in ("train", "valid"):
        os.makedirs(os.path.join(HEDEF, s, "images"))
        os.makedirs(os.path.join(HEDEF, s, "labels"))

    z = zipfile.ZipFile(ZIP)
    d = json.loads(z.read("Annotations/GBS_coco.json"))
    img = {i["id"]: i for i in d["images"]}
    kutular = {}
    for a in d["annotations"]:
        kutular.setdefault(a["image_id"], []).append(a)

    def gercek_orijinal(ad):
        return ad.lower().endswith(".jpg") and not ad.startswith(KOPYA_ONEKLERI)

    rnd = random.Random(42)
    sayac = {"poz_train": 0, "poz_valid": 0, "neg_train": 0, "neg_valid": 0}
    for iid in sorted(img):
        ad = img[iid]["file_name"]
        if not gercek_orijinal(ad):
            continue
        anns = kutular.get(iid, [])
        overflow = [a for a in anns if a["category_id"] == 0]
        bin_var = any(a["category_id"] == 1 for a in anns)
        if overflow:
            tur = "poz"
        elif bin_var:
            tur = "neg"
        else:
            continue
        split = "valid" if rnd.random() < 0.1 else "train"
        stem = "gbs_" + re.sub(r"\W", "_", os.path.splitext(ad)[0])
        if not yaz(z, ad, stem, split):
            continue
        W, H = img[iid]["width"], img[iid]["height"]
        satirlar = []
        for a in overflow:
            x, y, w, h = a["bbox"]
            xc, yc = (x + w / 2) / W, (y + h / 2) / H
            satirlar.append(f"0 {min(max(xc,0),1):.6f} {min(max(yc,0),1):.6f} "
                            f"{min(w/W,1):.6f} {min(h/H,1):.6f}")
        with open(os.path.join(HEDEF, split, "labels", stem + ".txt"), "w") as f:
            f.write(("\n".join(satirlar) + "\n") if satirlar else "")
        sayac[f"{tur}_{split}"] += 1
    print(sayac)


if __name__ == "__main__":
    main()
