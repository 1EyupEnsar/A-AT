r"""
merge_dataset.py'daki AYNI kaynaklari kullanarak, TEK bir 3-sinifli veri
seti yerine 3 AYRI veri seti olusturur:
  dataset/model_cukur/{train,valid}   (nc=1, Cukur)
  dataset/model_tabela/{train,valid}  (nc=1, HasarliTabela)
  dataset/model_kasis/{train,valid}   (nc=2, KasisBozuk + YayaGecidi)

Amac: 3 sinifi TEK kucuk modelde (yolov8n) birlestirince ortaya cikan
"paylasilan kapasite rekabeti" sorununu (bkz. rapor Bolum 6.4 - Cukur
%94.1(yalniz) -> %90.8(2 sinif) -> %83.8(3 sinif) seklinde duzenli
dusuyordu) ortadan kaldirmak icin her sinifi kendi UZMAN modelinde,
digerlerinden habersiz sekilde egitmek.

main.py bu modelleri ayni anda yukleyip her karede sirayla calistiracak.

NOT (2026-09-30): Disk yeri acmak icin CUKUR ham kaynak klasorleri
(cukur_universe, cukur_seg_universe, set1_universe, set2_universe,
rdd_universe, smartathon_pothole_universe, pothole_v1_universe, combined,
hard_negatives, logar_negatifleri) OneDrive'a tasindi:
C:\Users\lasto\OneDrive\Desktop\OneDrive\ASAT_proje_arsivi\dataset_universe_kaynaklari
Cukur icin YENIDEN build_cukur() calistirmadan once bu klasorleri geri
tasi, yoksa o kaynaklar SESSIZCE 0 goruntu katkisi yapar (copy_source hata
vermez, sadece atlar). TABELA kaynaklari (hasarli_tabela_universe,
tabela2/3_universe, road_inspection_universe, signs13/9/merged_universe,
smartathon_theme1_universe) YERİNDE birakildi.

NOT (2026-10-01): TasanCopKutusu (cop_kutusu_universe, cop_overflow_universe,
cop_binfull_universe, cop_hard_negatifleri, gbs_universe - hepsi OneDrive
arsivinde) TAMAMEN KALDIRILDI, yerine Kasis/YayaGecidi eklendi (bkz.
config.py Madde 10). Bu oturumda sadece build_kasis() calistirildi - Cukur
ve Tabela kasitli olarak DOKUNULMADI (main() TUMUNU yeniden kurar, bu yuzden
sadece Kasis guncellemek icin `import merge_dataset_separate as mds;
mds.build_kasis()` kullan, main() DEGIL).

NOT (2026-10-01, devam): model_kasis BASARIYLA OLUSTURULDUKTAN SONRA, KASIS_
POTHOLE_SRC (kasis_pothole_universe) ve CDSET_SRC (cdset_raw) klasorleri de
disk yeri acmak icin OneDrive'a tasindi (ayni ASAT_proje_arsivi/dataset_
universe_kaynaklari klasoru). build_kasis() TEKRAR calistirilmadan once bu
iki klasoru de geri tasi, yoksa Kasis veri seti SESSIZCE bos/eksik olusur.
"""
import os
import shutil

BASE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE, "dataset")

CUKUR_SRC = os.path.join(DATASET_DIR, "cukur_universe")
SET2_SRC = os.path.join(DATASET_DIR, "set2_universe")
RDD_SRC = os.path.join(DATASET_DIR, "rdd_universe")
SET1_SRC = os.path.join(DATASET_DIR, "set1_universe")
LOGAR_NEG_SRC = os.path.join(DATASET_DIR, "logar_negatifleri")
SMARTATHON_SRC = os.path.join(DATASET_DIR, "smartathon_pothole_universe")

# Kasis (hiz kesici) + Yaya Gecidi icin (2026-10-01, bkz. config.py Madde 10 -
# TasanCopKutusu'nun yerine eklendi). KASIS_POTHOLE_SRC: Roboflow
# "Humps/Bumps & Potholes Detection" (nc=2: Pothole=0, Speed Bump=1) -
# SADECE Speed Bump(1) alinir, Pothole(0) ATILIR (Cukur zaten kendi modelinde
# kapsaniyor, burada tekrar etmeye gerek yok). CDSET_SRC: UBI/Zenodo CDSet
# (nc=2: crosswalk=0, guide_arrows=0) - SADECE crosswalk(0) alinir; bu veri
# setinde bozuk/saglam ayrimi YOK (ham konum etiketi), bozukluk durumu
# INFERENCE SIRASINDA gecit_analiz.py ile belirlenir (bkz. config.py).
KASIS_POTHOLE_SRC = os.path.join(DATASET_DIR, "kasis_pothole_universe")
CDSET_SRC = os.path.join(DATASET_DIR, "cdset_raw", "CDSet", "dataset_YOLO_format_3434")

TABELA_SOURCES = [
    {"dir": "hasarli_tabela_universe", "prefix": "tabela1", "keep_ids": {0, 1}},
    {"dir": "tabela2_universe", "prefix": "tabela2", "keep_ids": {0, 3}},
    {"dir": "tabela3_universe", "prefix": "tabela3", "keep_ids": {0}},
    {"dir": "road_inspection_universe", "prefix": "roadinsp", "keep_ids": {0}},
    # Hasar siniflari tutuldu; 'ok', 'not-damaged', 'dirty', 'occluded', 'other'
    # hasar sayilmadigi icin ATILDI (o goruntuler etiketsiz negatif kalir).
    # signs13: cracked0 deformation1 faded3 graffiti4 knocked5 peeled9
    #          perforation10 rust11 stickers12
    {"dir": "signs13_universe", "prefix": "sg13", "keep_ids": {0, 1, 3, 4, 5, 9, 10, 11, 12}},
    # signs9: deformation0 graffiti2 knocked3 perforation6 stickers7 worn8
    {"dir": "signs9_universe", "prefix": "sg9", "keep_ids": {0, 2, 3, 6, 7, 8}},
    # merged: bent0 broken-sheet1 crack2 graffiti3 rust5
    {"dir": "signs_merged_universe", "prefix": "sgm", "keep_ids": {0, 1, 2, 3, 5}},
    # Smartathon Theme 1: BROKEN_SIGNAGE=2, FADED_SIGNAGE=5
    {"dir": "smartathon_theme1_universe", "prefix": "smt1", "keep_ids": {2, 5}},
]

# NOT (2026-10-01): COP_SOURCES ve TasanCopKutusu modeli TAMAMEN KALDIRILDI
# (bkz. config.py Madde 10). Kaynak veri setleri (cop1_/cop2_/cop3_/gbs_)
# gercek Turkiye sokak/dashcam goruntusuyle alakasiz bulundu. Yerine KASIS
# ve YAYA GECIDI eklendi (bkz. KASIS_POTHOLE_SRC, CDSET_SRC, build_kasis()).


import cv2

# Ayni goruntunun farkli kaynaklarda (orn. Smartathon'un iki seti, Damaged
# Traffic Signs'in iki surumu) tekrar sayilmasini ve train/valid arasinda
# sizmasini onlemek icin icerik tabanli (dhash) tekrar eleme. Roboflow farkli
# boyutlarda yeniden orneklediginden bayt esitligi yetmez.
SEEN = {}
DEDUPED = {}


def dhash(path):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    small = cv2.resize(im, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def reset_dir(out_dir):
    SEEN[out_dir] = set()
    DEDUPED[out_dir] = 0
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
            if len(parts) > 5:
                # Poligon satiri (segmentasyon etiketi): kutu modelleri icin
                # sinir kutusuna cevir; karisik satirlar YOLO'da goruntuyu dusurur.
                xs = [float(v) for v in parts[1::2]]
                ys = [float(v) for v in parts[2::2]]
                x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
                parts = [parts[0], f"{(x1 + x2) / 2:.6f}", f"{(y1 + y2) / 2:.6f}",
                         f"{x2 - x1:.6f}", f"{y2 - y1:.6f}"]
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
        h = dhash(os.path.join(img_dir, fname))
        if h is not None and out_dir in SEEN:
            if h in SEEN[out_dir]:
                DEDUPED[out_dir] += 1
                continue
            SEEN[out_dir].add(h)
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


def convert_label_multi(src_lbl_path, id_map):
    """convert_label ile AYNI mantik, ama TEK sinife (0) zorlamak yerine
    her kaynak sinif id'sini id_map[kaynak_id] -> cikti_id eslemesiyle
    cevirir (id_map'te olmayanlar atilir). build_kasis() gibi COK SINIFLI
    veri setleri icin kullanilir (bkz. Kasis: KasisBozuk=0, YayaGecidi=1)."""
    lines_out = []
    with open(src_lbl_path, "r") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            parts = raw.split()
            cls_id = int(float(parts[0]))
            if cls_id not in id_map:
                continue
            if len(parts) > 5:
                xs = [float(v) for v in parts[1::2]]
                ys = [float(v) for v in parts[2::2]]
                x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
                parts = [parts[0], f"{(x1 + x2) / 2:.6f}", f"{(y1 + y2) / 2:.6f}",
                         f"{x2 - x1:.6f}", f"{y2 - y1:.6f}"]
            parts[0] = str(id_map[cls_id])
            lines_out.append(" ".join(parts))
    return lines_out


def copy_source_multi(src_dir, prefix, id_map, split, out_dir, out_split):
    """copy_source ile ayni, ama convert_label_multi kullanir (cok sinifli)."""
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
        new_lines = convert_label_multi(src_lbl, id_map)
        h = dhash(os.path.join(img_dir, fname))
        if h is not None and out_dir in SEEN:
            if h in SEEN[out_dir]:
                DEDUPED[out_dir] += 1
                continue
            SEEN[out_dir].add(h)
        new_stem = f"{prefix}_{stem}"
        shutil.copy2(os.path.join(img_dir, fname), os.path.join(out_img_dir, new_stem + ext))
        with open(os.path.join(out_lbl_dir, new_stem + ".txt"), "w") as f:
            f.write("\n".join(new_lines))
            if new_lines:
                f.write("\n")
        count += 1
    return count


def copy_source_multi_altlayout(src_dir, prefix, id_map, split, out_dir, out_split):
    """copy_source_multi ile AYNI, ama CDSet gibi images/<split>+labels/<split>
    DIS duzeninde (digerlerinde <split>/images+<split>/labels - split ICERDE)
    olan kaynaklar icin. src_dir/images/split ve src_dir/labels/split okunur."""
    img_dir = os.path.join(src_dir, "images", split)
    lbl_dir = os.path.join(src_dir, "labels", split)
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
        new_lines = convert_label_multi(src_lbl, id_map)
        h = dhash(os.path.join(img_dir, fname))
        if h is not None and out_dir in SEEN:
            if h in SEEN[out_dir]:
                DEDUPED[out_dir] += 1
                continue
            SEEN[out_dir].add(h)
        new_stem = f"{prefix}_{stem}"
        shutil.copy2(os.path.join(img_dir, fname), os.path.join(out_img_dir, new_stem + ext))
        with open(os.path.join(out_lbl_dir, new_stem + ".txt"), "w") as f:
            f.write("\n".join(new_lines))
            if new_lines:
                f.write("\n")
        count += 1
    return count


def write_data_yaml_multi(out_dir, class_names):
    names_str = ", ".join(f"'{n}'" for n in class_names)
    content = (
        "train: train/images\n"
        "val: valid/images\n"
        "\n"
        f"nc: {len(class_names)}\n"
        f"names: [{names_str}]\n"
    )
    with open(os.path.join(out_dir, "data.yaml"), "w") as f:
        f.write(content)


def build_kasis():
    """Kasis modeli icin COK SINIFLI veri seti: KasisBozuk=0, YayaGecidi=1
    (bkz. config.py Madde 10, MODEL_SINIF_ADLARI). TasanCopKutusu'nun
    yerini alan yeni model."""
    out_dir = os.path.join(DATASET_DIR, "model_kasis")
    reset_dir(out_dir)
    tr, va = 0, 0
    # Roboflow "Humps/Bumps & Potholes Detection": Pothole=0 (ATILIR, Cukur
    # kendi modelinde kapsiyor), Speed Bump=1 -> bizim KasisBozuk=0.
    if os.path.isdir(KASIS_POTHOLE_SRC):
        id_map = {1: 0}
        tr += copy_source_multi(KASIS_POTHOLE_SRC, "kasis", id_map, "train", out_dir, "train")
        tr += copy_source_multi(KASIS_POTHOLE_SRC, "kasis", id_map, "test", out_dir, "train")
        va += copy_source_multi(KASIS_POTHOLE_SRC, "kasis", id_map, "valid", out_dir, "valid")
    # CDSet: crosswalk=0 -> bizim YayaGecidi=1, guide_arrows=1 ATILIR (bizim
    # kapsamimizda degil). CDSet kendi train/test ayrimini kullaniyor (valid
    # klasoru yok); test split'i bizim valid'imiz olarak kullanilir.
    if os.path.isdir(CDSET_SRC):
        id_map = {0: 1}
        tr += copy_source_multi_altlayout(CDSET_SRC, "cdset", id_map, "train", out_dir, "train")
        va += copy_source_multi_altlayout(CDSET_SRC, "cdset", id_map, "test", out_dir, "valid")
    # BagcılarYol'da KasisBozuk'un parke tasi/islak zemin/lastik izini kasis
    # sandigi tespit edildi (bkz. rapor Bolum 12.9.2, kasis_negatif_cikar.py).
    # Bu goruntuler ETIKETSIZ negatif olarak eklenir - model "bu kasis degil"
    # diye ogrensin.
    kasis_neg_dir = os.path.join(DATASET_DIR, "kasis_negatifleri")
    if os.path.isdir(kasis_neg_dir):
        tr += copy_as_negative(kasis_neg_dir, "bgclneg", "train", out_dir, "train")
        va += copy_as_negative(kasis_neg_dir, "bgclneg", "valid", out_dir, "valid")
    write_data_yaml_multi(out_dir, ["KasisBozuk", "YayaGecidi"])
    return tr, va


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
    # set2: gercek dashcam cukur/catlak/rogar kapagi (prepare_set2.py). Sadece
    # Pothole(0) tutulur; catlak/rogar kapagi olan ama cukur OLMAYAN goruntuler
    # bos etiketle "zor negatif" olarak kalir (rogar kapagi cukura benzer).
    if os.path.isdir(SET2_SRC):
        tr += copy_source(SET2_SRC, "set2", {0}, "train", out_dir, "train")
        va += copy_source(SET2_SRC, "set2", {0}, "valid", out_dir, "valid")
    # RDD2022 (prepare_rdd.py, sadece D40 iceren goruntuler) ve Smartathon
    # "New pothole detection" (tek sinif Pothole, Public Domain).
    # set1 'good' yol goruntuleri: etiketsiz negatif (prepare_set1.py).
    if os.path.isdir(SET1_SRC):
        tr += copy_source(SET1_SRC, "set1", {0}, "train", out_dir, "train")
        va += copy_source(SET1_SRC, "set1", {0}, "valid", out_dir, "valid")
    # Bayrampasa'da yanlis pozitif veren logar/rogar kapaklari: etiketsiz
    # negatif (logar_negatif_cikar.py - bkz. config.py'deki dairesellik notu).
    if os.path.isdir(LOGAR_NEG_SRC):
        tr += copy_source(LOGAR_NEG_SRC, "logarneg", {0}, "train", out_dir, "train")
        va += copy_source(LOGAR_NEG_SRC, "logarneg", {0}, "valid", out_dir, "valid")
    # Smartathon Theme 1 (POTHOLES=8) ve pothole_v1 (tek sinif) de eklenir;
    # ayni goruntuler dhash ile elenir.
    for src_dir, prefix, keep in ((RDD_SRC, "rdd", {0}), (SMARTATHON_SRC, "smth", {0}),
                                  (os.path.join(DATASET_DIR, "smartathon_theme1_universe"), "smt1", {8}),
                                  (os.path.join(DATASET_DIR, "pothole_v1_universe"), "phv1", {0})):
        if os.path.isdir(src_dir):
            tr += copy_source(src_dir, prefix, keep, "train", out_dir, "train")
            tr += copy_source(src_dir, prefix, keep, "test", out_dir, "train")
            va += copy_source(src_dir, prefix, keep, "valid", out_dir, "valid")
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
    k_tr, k_va = build_kasis()

    # Capraz negatif ornekler: Kasis/YayaGecidi, kompakt cukur/logar
    # seklindeki nesnelerle KARISMASIN diye (ayrica bkz. config.py
    # MIN_EN_BOY_ORANLARI - genislik/yukseklik >= 1.8 geometrik filtresi),
    # cukur kaynagini Kasis'e, Kasis/gecit kaynaklarini da Cukur'a etiketsiz
    # negatif olarak ekliyoruz.
    kasis_out = os.path.join(DATASET_DIR, "model_kasis")
    cukur_out = os.path.join(DATASET_DIR, "model_cukur")

    k_tr += copy_as_negative(CUKUR_SRC, "cukur", "train", kasis_out, "train")
    k_va += copy_as_negative(CUKUR_SRC, "cukur", "valid", kasis_out, "valid")

    if os.path.isdir(SET2_SRC):
        k_tr += copy_as_negative(SET2_SRC, "set2", "train", kasis_out, "train")
        k_va += copy_as_negative(SET2_SRC, "set2", "valid", kasis_out, "valid")

    if os.path.isdir(KASIS_POTHOLE_SRC):
        c_tr += copy_as_negative(KASIS_POTHOLE_SRC, "kasis", "train", cukur_out, "train")
        c_va += copy_as_negative(KASIS_POTHOLE_SRC, "kasis", "valid", cukur_out, "valid")

    for d, n in DEDUPED.items():
        print(f"  tekrar elenen goruntu: {os.path.basename(d)} = {n}")
    print("3 ayri veri seti olusturuldu (capraz negatiflerle):")
    print(f"  model_cukur  -> train {c_tr:4d} / valid {c_va:4d}")
    print(f"  model_tabela -> train {t_tr:4d} / valid {t_va:4d}")
    print(f"  model_kasis  -> train {k_tr:4d} / valid {k_va:4d}  (KasisBozuk+YayaGecidi)")


if __name__ == "__main__":
    main()
