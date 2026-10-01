"""
set1.zip (sih_road_dataset: yol kalitesi siniflandirma, kutu etiketi YOK) ->
dataset/set1_universe. Sadece 'good' klasoru alinir ve ETIKETSIZ negatif
olarak kullanilir (cukur/cop modellerinde 'burada hedef yok' ornegi).
'satisfactory/poor/very_poor' ALINMAZ: etiketsiz cukur icerebilirler, negatif
diye vermek modele yanlis ogretir.
"""
import os
import random
import shutil
import zipfile

BASE = os.path.dirname(os.path.abspath(__file__))
ZIP = r"C:\Users\lasto\OneDrive\Desktop\OneDrive\ASAT_veri_setleri_zip\set1.zip"
HEDEF = os.path.join(BASE, "dataset", "set1_universe")


def main():
    if os.path.exists(HEDEF):
        shutil.rmtree(HEDEF)
    for s in ("train", "valid"):
        os.makedirs(os.path.join(HEDEF, s, "images"))
        os.makedirs(os.path.join(HEDEF, s, "labels"))
    z = zipfile.ZipFile(ZIP)
    adlar = sorted(n for n in z.namelist()
                   if n.startswith("sih_road_dataset/good/") and n.lower().endswith((".jpg", ".jpeg", ".png")))
    rnd = random.Random(42)
    say = {"train": 0, "valid": 0}
    for n in adlar:
        split = "valid" if rnd.random() < 0.1 else "train"
        stem = os.path.splitext(os.path.basename(n))[0]
        with open(os.path.join(HEDEF, split, "images", stem + ".jpg"), "wb") as f:
            f.write(z.read(n))
        open(os.path.join(HEDEF, split, "labels", stem + ".txt"), "w").close()
        say[split] += 1
    print(say)


if __name__ == "__main__":
    main()
