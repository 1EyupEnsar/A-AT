"""
Akilli Sehir Ariza Tespit Sistemi - Python tespit servisi ayarlari.
Bu dosyadaki degerleri kendi ortamina gore duzenle.
"""
import os

# --- Yollar -----------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Her ariza turu icin AYRI, o turu uzmanlasmis sekilde ogrenmis bir model
# (bkz. merge_dataset_separate.py + train_separate.py). Tek bir 3 sinifli
# modelde birlestirmek yerine 3 ayri uzman modele gecildi, cunku kucuk
# yolov8n modeli 3 sinifi birden ogrenirken "paylasilan kapasite rekabeti"
# yasiyordu (bkz. rapor Bolum 6.4): Cukur mAP50 tek basina %94.1 iken 3
# sinif birlikte egitilince %83.8'e dusmustu. Ayri modellerle:
#   Cukur %94.1, HasarliTabela %69.9, TasanCopKutusu %86.4
# Bedeli: kare basina islem suresi ~13.6ms -> ~39ms (3 kat), ama bu hala
# standart bir dashcam'in (24-30 fps) gercek zamanli ihtiyacini rahatca
# karsiliyor (saniyede ~25.7 kare islenebiliyor, olcum yapildi).
MODEL_PATHS = {
    "Cukur": os.path.join(BASE_DIR, "models", "best_cukur.pt"),
    "HasarliTabela": os.path.join(BASE_DIR, "models", "best_tabela.pt"),
    "TasanCopKutusu": os.path.join(BASE_DIR, "models", "best_cop.pt"),
}

# Analiz edilecek video dosyasi. Canli kamera icin 0 (int) da verilebilir.
VIDEO_PATH = os.path.join(BASE_DIR, "videos", "gemini_test.mp4")

# ASP.NET Core backend'in calistigi adres (varsayilan Kestrel http portu).
API_BASE_URL = "http://localhost:5080"
API_ENDPOINT = f"{API_BASE_URL}/api/ariza"

# Tespit edilen karelerin kirpilip kaydedilecegi klasor. Backend projesinin
# wwwroot/uploads klasorunu gosterir ki API'ye gonderilen fotografYolu
# ("/uploads/...") backend tarafindan statik olarak servis edilebilsin.
# Bu depoyu tasidiysan ya da backend'i baska bir yere kurduysan burayi guncelle.
UPLOAD_DIR = os.path.join(
    BASE_DIR, "..", "backend", "ArizaTakipSistemi", "wwwroot", "uploads"
)

# --- Tespit ayarlari ----------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.5  # Bu esigin altindaki tespitler yok sayilir

# Yanlis pozitifleri (insan, motosikletli, arac gibi hareketli/dikine
# nesneleri) elemek icin basit bir geometrik kural: gercek cukurlar
# havadan/on kameradan bakildiginda genellikle enlemesine ya da kareye
# yakin gorunur (genislik >= yukseklik * bu oran); yurüyen bir insan ya da
# motosikletli surucu ise dikine, uzun bir siluet olusturur. Bu oranin
# altinda kalan tespitler "muhtemelen cukur degil" sayilip atlanir.
MIN_EN_BOY_ORANI = 0.7  # genislik / yukseklik

# MIN_EN_BOY_ORANI filtresi SADECE bu siniflara uygulanir. Cukurlar
# havadan/on kameradan enlemesine gorunur, ama hasarli levhalar dogal
# olarak dikine/uzun kutular olabilir (levha dik durur) - bu yuzden
# levhaya bu filtreyi uygulamak gercek tespitleri de elerdi.
EN_BOY_FILTRESI_UYGULANACAK_TURLER = {"Cukur"}

# --- Ariza siddeti (kaba alan tabanli tahmin) ---------------------------------
# Fikir: Kaggle'daki "Pothole Segmentation for Road Damage Assessment"
# projesinden esinlenildi - orada segmentasyon maskesinin piksel alani
# hesaplanip yol hasar yuzdesi cikariliyordu. Biz segmentasyona gecmeden,
# tespit KUTUSUNUN kare alanina orani ile kaba bir "kucuk/orta/buyuk"
# siddet tahmini yapiyoruz. Bu esikler gozlemsel/placeholder degerlerdir;
# gercek sahada toplanan verilerle ayarlanabilir. SADECE Cukur icin
# anlamli (bir levha ya da cop kutusunun "buyuklugu" ayni seyi ifade etmez).
SIDDET_UYGULANACAK_TURLER = {"Cukur"}
SIDDET_ESIK_KUCUK_ORTA = 0.015   # kare alaninin bu oranindan kucukse "Kucuk"
SIDDET_ESIK_ORTA_BUYUK = 0.04    # bu oranin ustundeyse "Buyuk", arasi "Orta"

# Aynı arıza türünün saniyede onlarca kez kaydedilip API'ye gonderilmesini
# onlemek icin sinif bazli soguma suresi (saniye).
COOLDOWN_SECONDS = 8.0

# Ayni fiziksel arizanin (orn. ayni cukur) video birden fazla kez izlenince
# ya da video ayni yoldan tekrar gecince TEKRAR TEKRAR kaydedilmesini onlemek
# icin: yeni bir tespit gonderilmeden once, backend'de AYNI turden, henuz
# "Bekliyor" durumunda (yani cozulmemis) ve bu koordinata bu kadar metreden
# yakin acik bir kayit var mi diye kontrol edilir; varsa yeni kayit ATLANIR.
# Gercek hayatta da mantikli: bir kamyonet ayni sokaktan tekrar gecerse ve
# ariza hala cozulmemisse, aynisini tekrar rapor etmeye gerek yok.
DUPLICATE_MESAFE_METRE = 20.0

# Video oynatim penceresini goster (True) / arka planda sessizce calistir (False)
SHOW_PREVIEW = True

# API'ye istek zaman asimi (saniye)
REQUEST_TIMEOUT = 5

# --- Konum (GPS) simulasyonu --------------------------------------------------
# Kamyonete gercek bir GPS modulu (orn. NMEA seri port ile okunan bir alici)
# takildiginda, main.py icindeki KonumSimulatoru sinifi yerine o modulden
# anlik enlem/boylam okuyan bir fonksiyon kullanilmalidir. Simdilik demo
# amacli, kamyonetin bir rota boyunca yavasca ilerledigini simule ediyoruz.
GPS_BASLANGIC_ENLEM = 41.0082    # Istanbul (ornek baslangic noktasi)
GPS_BASLANGIC_BOYLAM = 28.9784
GPS_ADIM_ENLEM = 0.00015         # her tespitte rota boyunca kucuk bir ilerleme
GPS_ADIM_BOYLAM = 0.00022
