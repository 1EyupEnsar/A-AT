"""
Sadece TRAIN klasorunu, etiketleri koruyarak (yatay ayna, kirpma-yakinlastirma,
parlaklik/gama/HSV, bulaniklik, hareket bulaniklik, gurultu, JPEG, sis, dusuk
isik, golge) cogaltir. VALID'e ASLA dokunmaz (sizinti olmasin).

Uretilen dosyalar 'aug<j>_' on ekiyle yazilir; script tekrar calisirsa eski
'aug*' dosyalari once silinir (ust uste binmez). Sayilar
dataset/augment_report.json'a yazilir - rapor icin GERCEK ve AUGMENTE
goruntuler ayri sayilir.

Kullanim: python augment_dataset.py model_cukur <poz_carpan> <neg_carpan>
"""
import json
import os
import random
import sys
from multiprocessing import Pool

import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(BASE, "dataset")


def oku_etiket(p):
    if not os.path.exists(p):
        return []
    out = []
    for l in open(p):
        s = l.split()
        if len(s) == 5:
            out.append([int(float(s[0]))] + [float(v) for v in s[1:]])
    return out


def yaz_etiket(p, kutular):
    with open(p, "w") as f:
        for c, x, y, w, h in kutular:
            f.write(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def fotometrik(im, r):
    ops = ["parlaklik", "gama", "hsv", "bulanik", "hareket", "gurultu", "jpeg", "sis", "karanlik", "golge"]
    for op in r.sample(ops, r.choice([2, 3])):
        if op == "parlaklik":
            im = cv2.convertScaleAbs(im, alpha=r.uniform(0.65, 1.4), beta=r.uniform(-35, 35))
        elif op == "gama":
            g = r.uniform(0.6, 1.6)
            lut = np.array([((i / 255.0) ** g) * 255 for i in range(256)], np.uint8)
            im = cv2.LUT(im, lut)
        elif op == "hsv":
            hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[..., 1] *= r.uniform(0.6, 1.4)
            hsv[..., 0] = (hsv[..., 0] + r.uniform(-8, 8)) % 180
            im = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        elif op == "bulanik":
            k = r.choice([3, 5, 7])
            im = cv2.GaussianBlur(im, (k, k), 0)
        elif op == "hareket":
            k = r.choice([5, 7, 9, 11])
            ker = np.zeros((k, k), np.float32)
            ker[k // 2, :] = 1.0 / k
            m = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), r.uniform(0, 180), 1)
            ker = cv2.warpAffine(ker, m, (k, k))
            s = ker.sum()
            im = cv2.filter2D(im, -1, ker / s) if s > 0 else im
        elif op == "gurultu":
            n = np.random.RandomState(r.randint(0, 2**31 - 1)).normal(0, r.uniform(4, 14), im.shape)
            im = np.clip(im.astype(np.float32) + n, 0, 255).astype(np.uint8)
        elif op == "jpeg":
            _, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, r.randint(25, 60)])
            im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        elif op == "sis":
            a = r.uniform(0.15, 0.4)
            im = cv2.addWeighted(im, 1 - a, np.full_like(im, 225), a, 0)
        elif op == "karanlik":
            im = np.clip(im.astype(np.float32) * r.uniform(0.4, 0.7), 0, 255).astype(np.uint8)
        elif op == "golge":
            h, w = im.shape[:2]
            pts = np.array([[r.randint(0, w), r.randint(0, h)] for _ in range(4)], np.int32)
            mask = np.zeros((h, w), np.uint8)
            cv2.fillPoly(mask, [pts], 255)
            f = r.uniform(0.5, 0.8)
            im = np.where(mask[..., None] > 0, (im * f).astype(np.uint8), im)
    return im


def kirp_yakinlastir(im, kutular, r):
    h, w = im.shape[:2]
    s = r.uniform(0.65, 0.95)
    cw, ch = int(w * s), int(h * s)
    x0, y0 = r.randint(0, w - cw), r.randint(0, h - ch)
    yeni = []
    for c, x, y, bw, bh in kutular:
        ax1, ay1 = (x - bw / 2) * w, (y - bh / 2) * h
        ax2, ay2 = (x + bw / 2) * w, (y + bh / 2) * h
        ix1, iy1, ix2, iy2 = max(ax1, x0), max(ay1, y0), min(ax2, x0 + cw), min(ay2, y0 + ch)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        if (ix2 - ix1) * (iy2 - iy1) < 0.5 * (ax2 - ax1) * (ay2 - ay1):
            continue
        yeni.append([c, ((ix1 + ix2) / 2 - x0) / cw, ((iy1 + iy2) / 2 - y0) / ch,
                     (ix2 - ix1) / cw, (iy2 - iy1) / ch])
    if kutular and not yeni:
        return None
    return cv2.resize(im[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_LINEAR), yeni


def isle(args):
    img_yol, lbl_yol, out_img, out_lbl, stem, k, seed = args
    r = random.Random(seed)
    im = cv2.imread(img_yol)
    if im is None:
        return 0
    kutular = oku_etiket(lbl_yol)
    yazilan = 0
    for j in range(k):
        v, kb = im.copy(), [b[:] for b in kutular]
        if r.random() < 0.5:
            v = cv2.flip(v, 1)
            for b in kb:
                b[1] = 1 - b[1]
        if r.random() < 0.4:
            sonuc = kirp_yakinlastir(v, kb, r)
            if sonuc is not None:
                v, kb = sonuc
        v = fotometrik(v, r)
        ad = f"aug{j}_{stem}"
        cv2.imwrite(os.path.join(out_img, ad + ".jpg"), v, [cv2.IMWRITE_JPEG_QUALITY, 92])
        yaz_etiket(os.path.join(out_lbl, ad + ".txt"), kb)
        yazilan += 1
    return yazilan


def main():
    model, poz_m, neg_m = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    img_dir = os.path.join(DATASET, model, "train", "images")
    lbl_dir = os.path.join(DATASET, model, "train", "labels")
    for d in (img_dir, lbl_dir):
        for f in os.listdir(d):
            if f.startswith("aug"):
                os.remove(os.path.join(d, f))

    gercek_poz = gercek_neg = 0
    isler = []
    rnd = random.Random(1234)
    for f in sorted(os.listdir(img_dir)):
        stem = os.path.splitext(f)[0]
        lbl = os.path.join(lbl_dir, stem + ".txt")
        pozitif = os.path.exists(lbl) and os.path.getsize(lbl) > 0
        m = poz_m if pozitif else neg_m
        k = int(m) + (1 if rnd.random() < m - int(m) else 0)
        if pozitif:
            gercek_poz += 1
        else:
            gercek_neg += 1
        if k:
            isler.append((os.path.join(img_dir, f), lbl, img_dir, lbl_dir, stem, k, rnd.randint(0, 2**31 - 1)))

    with Pool(2) as p:
        aug = sum(p.imap_unordered(isle, isler, chunksize=64))

    valid_say = len(os.listdir(os.path.join(DATASET, model, "valid", "images")))
    rapor_yol = os.path.join(DATASET, "augment_report.json")
    rapor = json.load(open(rapor_yol)) if os.path.exists(rapor_yol) else {}
    rapor[model] = {
        "gercek_train_pozitif": gercek_poz, "gercek_train_negatif": gercek_neg,
        "gercek_valid": valid_say, "augmente_train": aug,
        "toplam": gercek_poz + gercek_neg + valid_say + aug,
    }
    json.dump(rapor, open(rapor_yol, "w"), indent=2)
    print(model, rapor[model])


if __name__ == "__main__":
    main()
