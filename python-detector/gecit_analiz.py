"""Yaya gecidi cizgi gorunurlugu analizi (klasik goruntu isleme) - v2.

YOLO modeli bir bolgeyi "YayaGecidi" (konum) olarak isaretledikten SONRA, bu
modul o bolgedeki cizgilerin GORUNUP GORUNMEDIGINI olcer. main.py bu skora
gore ya hic raporlamaz (saglam/belirsiz -> ariza degil) ya da "GecitKullanilamaz"
olarak isaretler (cizgiler acikca kaybolmus).

TASARIM ILKESI (2026-10-01, kullanicinin acik talebi): "cok temkinli olsun,
ama asiri bozuk olani yakalasin" - yani YANLIS NEGATIF (gercek bozuk bir
gecidi kacirmak) kabul edilebilir, YANLIS POZITIF (saglam bir gecidi bozuk
sanmak) KABUL EDILEMEZ. Bu yuzden tek, basit ve yoruma acik olmayan bir
metrik kullanilir: kirpilan bolgedeki BEYAZ PIKSEL ORANI (Otsu esikleme).
Saglam/orta-asinmis bir gecit her zaman belirgin miktarda beyaz boya icerir;
sadece boya neredeyse tamamen silinmisse oran kritik derecede dusuk olur.

ONCEKI SURUM (v1, Canny+Hough Line Transform tabanli) REDDEDILDI: BagcilarYol
testinde cok net/saglam bir gecidi (kare 255) yanlislikla "kullanilamaz" diye
isaretledi. Kok neden: YOLO kutusu siklikla onde giden aracin tampon kismini
da iceriyor; Canny bu guclu/keskin tampon kenarini BASKIN bulup gercek (daha
dusuk kontrastli ama olculen ~90 birimlik gercek bir farkli) serit kenarlarini
golgede birakiyordu. v2 bu sorunu iki sekilde cozer: (1) kutunun sadece ALT
kismi (yol yuzeyi, aracin govdesinin olmasi beklenmeyen bolge) kullanilir,
(2) kenar/cizgi GEOMETRISI yerine basit RENK ORANI olculur - aracin metal/
cam govdesi Otsu esiklemede "beyaz" olarak sayilsa bile, sadece alt kismi
kullanmak bu riski buyuk olcude azaltir.
"""
import cv2
import numpy as np

# Kutunun sadece bu kadarlik ALT kismi kullanilir (ust kisimda sik sik onde
# giden bir arac govdesi oluyor - bkz. yukaridaki not).
ALT_KISIM_ORANI = 0.55


def cizgi_gorunurlugu_skoru(kirpik_bgr) -> float:
    """0.0 (beyaz boya yok/neredeyse yok) ile 1.0 (yogun beyaz boya) arasi
    bir skor dondurur. Girdi: bir YayaGecidi tespitinin kirpilmis BGR
    goruntusu (tam kutu, kirpma bu fonksiyon icinde yapilir)."""
    if kirpik_bgr is None or kirpik_bgr.size == 0:
        return 1.0  # bilgi yok -> temkinli davran, "saglam" varsay (raporlama)

    h, w = kirpik_bgr.shape[:2]
    if h < 6 or w < 6:
        return 1.0

    alt = kirpik_bgr[int(h * (1 - ALT_KISIM_ORANI)):, :]
    gri = cv2.cvtColor(alt, cv2.COLOR_BGR2GRAY)

    # Otsu esikleme: goruntuyu kendi icindeki en iyi ikili ayrima noktasina
    # gore boler (sabit bir parlaklik esigi degil - farkli isik/golge
    # kosullarina kendiliginden uyum saglar).
    _, ikili = cv2.threshold(gri, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    beyaz_oran = float(np.count_nonzero(ikili)) / ikili.size

    # Otsu her zaman goruntuyu IKIYE boler (en aydinlik grup "beyaz" sayilir),
    # bu yuzden duz/tek renkli bir yol goruntusunde bile oran ~0.5 cikabilir.
    # Gercek ayrim gucu, iki grup arasindaki KONTRASTA bakilarak dogrulanir:
    # kontrast dusukse (duz gri asfalt, boya yok) oran anlamsizdir, "boya var"
    # sayilmaz.
    ort_karanlik = gri[ikili == 0].mean() if np.any(ikili == 0) else 0.0
    ort_parlak = gri[ikili == 255].mean() if np.any(ikili == 255) else 255.0
    kontrast = ort_parlak - ort_karanlik
    if kontrast < 35:
        return 1.0  # net bir beyaz/gri ayrimi yok -> temkinli: "saglam" varsay

    return beyaz_oran
