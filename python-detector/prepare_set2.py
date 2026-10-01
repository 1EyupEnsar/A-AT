"""
set2.zip (Road Damage Dataset: Pothole=0, Crack=1, Manhole=2; 2009 gercek
dashcam goruntusu, kutu etiketli) -> dataset/set2_universe/{train,valid}
duzenine cevirir. Kaynakta train/valid ayrimi yok; sabit tohumla %90/%10
bolunur (tekrarlanabilir). Sinif filtreleme merge_dataset_separate.py'de.
"""
import os
import random
import shutil

BASE = os.path.dirname(os.path.abspath(__file__))
KAYNAK = os.path.join(BASE, "dataset", "set2_raw", "data")
HEDEF = os.path.join(BASE, "dataset", "set2_universe")


def main():
    if os.path.exists(HEDEF):
        shutil.rmtree(HEDEF)
    for s in ("train", "valid"):
        os.makedirs(os.path.join(HEDEF, s, "images"))
        os.makedirs(os.path.join(HEDEF, s, "labels"))

    stemler = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(KAYNAK, "images")))
    random.Random(42).shuffle(stemler)
    n_val = len(stemler) // 10
    sayac = {"train": 0, "valid": 0}
    for i, stem in enumerate(stemler):
        split = "valid" if i < n_val else "train"
        lbl = os.path.join(KAYNAK, "labels-YOLO", stem + ".txt")
        if not os.path.exists(lbl):
            continue
        shutil.copy2(os.path.join(KAYNAK, "images", stem + ".jpg"),
                     os.path.join(HEDEF, split, "images", stem + ".jpg"))
        shutil.copy2(lbl, os.path.join(HEDEF, split, "labels", stem + ".txt"))
        sayac[split] += 1
    print(sayac)


if __name__ == "__main__":
    main()
