"""
RDD2022 (Pascal VOC, ulke basina ic ice zip) -> dataset/rdd_universe/{train,valid}
(YOLO kutu formati). Sadece en az bir D40 (Pothole) iceren goruntuler alinir;
D40 disindaki hasar kutulari (catlak vb.) atilir, cunku Cukur tek sinifli bir
uzman model. Sabit tohumla %90/%10 bolunur.
"""
import glob
import os
import random
import re
import shutil
import zipfile

BASE = os.path.dirname(os.path.abspath(__file__))
KAYNAK = os.path.join(BASE, "dataset", "rdd2022_raw", "RDD2022")
HEDEF = os.path.join(BASE, "dataset", "rdd_universe")


def voc_to_yolo(xml_text):
    w = int(re.search(r"<width>(\d+)</width>", xml_text).group(1))
    h = int(re.search(r"<height>(\d+)</height>", xml_text).group(1))
    satirlar = []
    for obj in re.findall(r"<object>(.*?)</object>", xml_text, re.S):
        if re.search(r"<name>(.*?)</name>", obj).group(1) != "D40":
            continue
        xmin, ymin, xmax, ymax = (float(re.search(rf"<{k}>([\d.]+)</{k}>", obj).group(1))
                                  for k in ("xmin", "ymin", "xmax", "ymax"))
        xc, yc = (xmin + xmax) / 2 / w, (ymin + ymax) / 2 / h
        bw, bh = (xmax - xmin) / w, (ymax - ymin) / h
        satirlar.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return satirlar


def main():
    if os.path.exists(HEDEF):
        shutil.rmtree(HEDEF)
    for s in ("train", "valid"):
        os.makedirs(os.path.join(HEDEF, s, "images"))
        os.makedirs(os.path.join(HEDEF, s, "labels"))

    rnd = random.Random(42)
    toplam = {"train": 0, "valid": 0}
    for zp in sorted(glob.glob(os.path.join(KAYNAK, "*.zip"))):
        ulke = os.path.splitext(os.path.basename(zp))[0]
        z = zipfile.ZipFile(zp)
        adlar = {os.path.basename(n): n for n in z.namelist()}
        for xml_ad, xml_yol in sorted(adlar.items()):
            if not xml_ad.endswith(".xml"):
                continue
            satirlar = voc_to_yolo(z.read(xml_yol).decode("utf8", "ignore"))
            jpg_ad = xml_ad[:-4] + ".jpg"
            if not satirlar or jpg_ad not in adlar:
                continue
            split = "valid" if rnd.random() < 0.1 else "train"
            stem = f"rdd_{ulke}_{xml_ad[:-4]}"
            with open(os.path.join(HEDEF, split, "images", stem + ".jpg"), "wb") as f:
                f.write(z.read(adlar[jpg_ad]))
            with open(os.path.join(HEDEF, split, "labels", stem + ".txt"), "w") as f:
                f.write("\n".join(satirlar) + "\n")
            toplam[split] += 1
        print(ulke, toplam)
    print("bitti:", toplam)


if __name__ == "__main__":
    main()
