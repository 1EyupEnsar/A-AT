"""
merge_dataset.py'daki AYNI kaynaklari kullanarak, TEK bir 3-sinifli veri
seti yerine 3 AYRI tek-sinifli veri seti olusturur:
  dataset/model_cukur/{train,valid}   (nc=1, Cukur)
  dataset/model_tabela/{train,valid}  (nc=1, HasarliTabela)
  dataset/model_cop/{train,valid}     (nc=1, TasanCopKutusu)

Amac: 3 sinifi TEK kucuk modelde (yolov8n) birlestirince ortaya cikan
"paylasilan kapasite rekabeti" sorununu (bkz. rapor Bolum 6.4 - Cukur
%94.1(yalniz) -> %90.8(2 sinif) -> %83.8(3 sinif) seklinde duzenli
dusuyordu) ortadan kaldirmak icin her sinifi kendi UZMAN modelinde,
digerlerinden habersiz sekilde egitmek.

main.py bu 3 modeli ayni anda yukleyip her karede sirayla calistiracak.
"""
import os
import shutil

BASE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE, "dataset")

CUKUR_SRC = os.path.join(DATASET_DIR, "cukur_universe")

TABELA_SOURCES = [
    {"dir": "hasarli_tabela_universe", "prefix": "tabela1", "keep_ids": {0, 1}},
    {"dir": "tabela2_universe", "prefix": "tabela2", "keep_ids": {0, 3}},
    {"dir": "tabela3_universe", "prefix": "tabela3", "keep_ids": {0}},
    {"dir": "road_inspection_universe", "prefix": "roadinsp", "keep_ids": {0}},
]

COP_SOURCES = [
    {"dir": "cop_kutusu_universe", "prefix": "cop1", "keep_ids": {1, 3}},
]


def reset_dir(out_dir):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    for split in ("train", "valid"):
        os.makedirs(os.path.join(out_dir, split, "images"))
        os.makedirs(os.path.join(out_dir, split, "labels"))


def convert_label(src_lbl_path, keep_ids):
    lines_out = []
    with open(src_lbl_path, "r") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            parts = raw.split()
            cls_id = int(float(parts[0]))
            if cls_id not in keep_ids:
                continue
            parts[0] = "0"  # tek sinifli veri setinde hedef her zaman 0
            lines_out.append(" ".join(parts))
    return lines_out


def copy_source(src_dir, prefix, keep_ids, split, out_dir, out_split):
    img_dir = os.path.join(src_dir, split, "images")
    lbl_dir = os.path.join(src_dir, split, "labels")
    if not os.path.isdir(img_dir):
        return 0
    out_img_dir = os.path.join(out_dir, out_split, "images")
    out_lbl_dir = os.path.join(out_dir, out_split, "labels")

    count = 0
    for fname in os.listdir(img_dir):
        stem, ext = os.path.splitext(fname)
        src_lbl = os.path.join(lbl_dir, stem + ".txt")
        if not os.path.exists(src_lbl):
            continue
        new_lines = convert_label(src_lbl, keep_ids)
        new_stem = f"{prefix}_{stem}"
        shutil.copy2(os.path.join(img_dir, fname), os.path.join(out_img_dir, new_stem + ext))
        with open(os.path.join(out_lbl_dir, new_stem + ".txt"), "w") as f:
            f.write("\n".join(new_lines))
            if new_lines:
                f.write("\n")
        count += 1
    return count


def copy_as_negative(src_dir, prefix, split, out_dir, out_split):
    """Kaynaktaki goruntuleri, ETIKETSIZ (bos label dosyasi) olarak baska
    bir modelin veri setine 'bu SENIN sinifin degil' ornegi olarak ekler.
    Ayri uzman modeller birbirinin goruntusunu HIC gormedigi icin, bir
    sinifa gorsel olarak benzeyen baska bir arizayi (orn. cukurun karanlik
    kenar + acik orta deseni ile dolu cop kutusunun benzer deseni) yanlislikla
    kendi sinifi sanabiliyordu - bkz. rapor Bolum 6.6."""
    img_dir = os.path.join(src_dir, split, "images")
    if not os.path.isdir(img_dir):
        return 0
    out_img_dir = os.path.join(out_dir, out_split, "images")
    out_lbl_dir = os.path.join(out_dir, out_split, "labels")

    count = 0
    for fname in os.listdir(img_dir):
        stem, ext = os.path.splitext(fname)
        new_stem = f"{prefix}_neg_{stem}"
        shutil.copy2(os.path.join(img_dir, fname), os.path.join(out_img_dir, new_stem + ext))
        open(os.path.join(out_lbl_dir, new_stem + ".txt"), "w").close()  # bos etiket
        count += 1
    return count


def write_data_yaml(out_dir, class_name):
    content = (
        "train: train/images\n"
        "val: valid/images\n"
        "\n"
        "nc: 1\n"
        f"names: ['{class_name}']\n"
    )
    with open(os.path.join(out_dir, "data.yaml"), "w") as f:
        f.write(content)


def build_cukur():
    out_dir = os.path.join(DATASET_DIR, "model_cukur")
    reset_dir(out_dir)
    tr = copy_source(CUKUR_SRC, "cukur", {0}, "train", out_dir, "train")
    va = copy_source(CUKUR_SRC, "cukur", {0}, "valid", out_dir, "valid")
    write_data_yaml(out_dir, "Cukur")
    return tr, va


def build_from_sources(sources, out_name, class_name):
    out_dir = os.path.join(DATASET_DIR, out_name)
    reset_dir(out_dir)
    total_tr, total_va = 0, 0
    for src in sources:
        src_dir = os.path.join(DATASET_DIR, src["dir"])
        total_tr += copy_source(src_dir, src["prefix"], src["keep_ids"], "train", out_dir, "train")
        total_va += copy_source(src_dir, src["prefix"], src["keep_ids"], "valid", out_dir, "valid")
        total_tr += copy_source(src_dir, src["prefix"], src["keep_ids"], "test", out_dir, "train")
    write_data_yaml(out_dir, class_name)
    return total_tr, total_va


def main():
    c_tr, c_va = build_cukur()
    t_tr, t_va = build_from_sources(TABELA_SOURCES, "model_tabela", "HasarliTabela")
    k_tr, k_va = build_from_sources(COP_SOURCES, "model_cop", "TasanCopKutusu")

    # Capraz negatif ornekler: cukur <-> cop kutusu karisikligi tespit edildi
    # (gercek panelde cukur, TasanCopKutusu olarak kaydedilmisti). Her iki
    # modele de digerinin goruntulerini etiketsiz "bu senin sinifin degil"
    # ornegi olarak ekliyoruz.
    cop_out = os.path.join(DATASET_DIR, "model_cop")
    cukur_out = os.path.join(DATASET_DIR, "model_cukur")

    k_tr += copy_as_negative(CUKUR_SRC, "cukur", "train", cop_out, "train")
    k_va += copy_as_negative(CUKUR_SRC, "cukur", "valid", cop_out, "valid")

    cop_src_dir = os.path.join(DATASET_DIR, COP_SOURCES[0]["dir"])
    c_tr += copy_as_negative(cop_src_dir, "cop1", "train", cukur_out, "train")
    c_va += copy_as_negative(cop_src_dir, "cop1", "valid", cukur_out, "valid")

    print("3 ayri tek-sinifli veri seti olusturuldu (capraz negatiflerle):")
    print(f"  model_cukur  -> train {c_tr:4d} / valid {c_va:4d}")
    print(f"  model_tabela -> train {t_tr:4d} / valid {t_va:4d}")
    print(f"  model_cop    -> train {k_tr:4d} / valid {k_va:4d}")


if __name__ == "__main__":
    main()
