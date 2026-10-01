"""
Cukur (pothole) icin GERCEK PIKSEL SEGMENTASYONU veri setini hazirlar.

Kaynak: dataset/cukur_seg_universe/ (Roboflow'dan fork edilen, Instance
Segmentation formatinda, 3271 goruntu). Kaynak veri setinde AYNI kavram
("cukur") yanlislikla 2 farkli sinif ID'sine ("POTHOLE"=0, "pothole"=1)
bolunmus - muhtemelen orijinal etiketleyicilerin buyuk/kucuk harf
tutarsizligi. Bu script HER IKISINI de tek bir sinifa (Cukur=0) indirger.

Cikti: dataset/model_cukur_seg/{train,valid,test}/{images,labels} + data.yaml
Etiket formati degismez (poligon noktalari), sadece sinif ID'si 0'a sabitlenir.

main.py/web_video_test.py'deki severity (siddet) hesaplamasi, bu modelle
egitilince kutu-alani yerine GERCEK maske piksel alanini kullanabilecek
(bkz. gelecek adim: siddet_hesapla_segmentasyon).
"""
import os
import shutil

BASE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE, "dataset")

KAYNAK = os.path.join(DATASET_DIR, "cukur_seg_universe")
HEDEF = os.path.join(DATASET_DIR, "model_cukur_seg")


def reset_dir(out_dir):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    for split in ("train", "valid", "test"):
        os.makedirs(os.path.join(out_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, split, "labels"), exist_ok=True)


def etiketi_tek_sinifa_indirge(src_lbl_path):
    """Poligon satirlarinin ilk sutunundaki sinif ID'sini (0 ya da 1, ikisi
    de 'cukur' anlamina geliyor) daima 0'a sabitler, kalan poligon
    koordinatlarina dokunmaz."""
    lines_out = []
    with open(src_lbl_path, "r") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            parts = raw.split()
            parts[0] = "0"
            lines_out.append(" ".join(parts))
    return lines_out


def bir_split_isle(split):
    img_dir = os.path.join(KAYNAK, split, "images")
    lbl_dir = os.path.join(KAYNAK, split, "labels")
    if not os.path.isdir(img_dir):
        return 0

    out_img_dir = os.path.join(HEDEF, split, "images")
    out_lbl_dir = os.path.join(HEDEF, split, "labels")

    count = 0
    for fname in os.listdir(img_dir):
        stem, ext = os.path.splitext(fname)
        src_lbl = os.path.join(lbl_dir, stem + ".txt")
        if not os.path.exists(src_lbl):
            continue
        yeni_satirlar = etiketi_tek_sinifa_indirge(src_lbl)
        shutil.copy2(os.path.join(img_dir, fname), os.path.join(out_img_dir, fname))
        with open(os.path.join(out_lbl_dir, stem + ".txt"), "w") as f:
            f.write("\n".join(yeni_satirlar))
            if yeni_satirlar:
                f.write("\n")
        count += 1
    return count


def write_data_yaml():
    content = (
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        "\n"
        "nc: 1\n"
        "names: ['Cukur']\n"
    )
    with open(os.path.join(HEDEF, "data.yaml"), "w") as f:
        f.write(content)


def main():
    reset_dir(HEDEF)
    tr = bir_split_isle("train")
    va = bir_split_isle("valid")
    te = bir_split_isle("test")
    write_data_yaml()

    print("Segmentasyon veri seti hazirlandi (2 sinif -> 1 sinife indirgendi):")
    print(f"  train: {tr}")
    print(f"  valid: {va}")
    print(f"  test:  {te}")
    print(f"  Toplam: {tr + va + te}")


if __name__ == "__main__":
    main()
