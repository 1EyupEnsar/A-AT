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
# Madde 8: Cukur artik bir DEDEKSIYON modeli degil, bir SEGMENTASYON modeli
# (yolov8n-seg) - gercek piksel maskesi urettigi icin siddet_hesapla_maskeden
# fonksiyonu kutu-alani yerine GERCEK maske alanini kullanabiliyor. Kendi
# temiz veri setimizde (94,1% mAP50) degil, cok daha buyuk/cesitli 3271
# goruntuluk yeni bir Roboflow veri setinde egitildi (78,9% box mAP50); kendi
# test videomuzda karsilastirildi: eski kutu modeline gore ~%16 daha az
# tespit (750->630) ama karsiliginda gercek segmentasyon yetenegi kazanildi -
# kabul edilebilir bir odunlesim (bkz. Bolum 7.3'teki emsal).
#
# Madde 9: TasanCopKutusu icin 2 yeni Roboflow kaynagindan (+184 goruntu,
# bin-empty sinifi bilincli disarida birakildi - bkz. merge_dataset_separate.py)
# egitilen v3 modeli, ILK BAKISTA agregat val metriklerinde v2'den DAHA KOTU
# gorunuyordu (mAP50 %72,0->%64,6, recall %71,2->%56,2). Bu dusus ONCE bir
# regresyon sanildi, ama arastirma bunun bir OLCUM ARTEFAKTI oldugunu ortaya
# cikardi: yeni val setine eklenen 43 goruntu (cop2_/cop3_ on ekli, farkli
# gorunumlu "tasan/dolu cop kutusu" kaynaklari) eski sete gore cok daha zor,
# agregat metrigi asagi cekiyor. Orijinal kaynaktaki 98 goruntuyu (27 gercek
# pozitif + 71 gercek negatif) izole edip AYNI eski gorevde v2 ve v3'u ayri
# ayri test edince: v2 recall %44,4 (12/27) -> v3 recall %88,9 (24/27), yanlis
# pozitif orani neredeyse ayni (71 negatifte 1 -> 2). Yani v3, orijinal
# gorevde bile v2'nin NEREDEYSE IKI KATI kadar iyi - agregat metrik dususu
# yaniltici bir benchmark-zorlugu artefaktiydi, gercek bir kotulesme degildi.
# v3 uretime alindi (best_cop.pt), eski v2 best_cop_v2_arsiv.pt olarak
# referans/geri-donus icin saklandi (bkz. rapor Bolum 7.4).
#
# Madde 9b: Gercek Bayrampasa dashcam videosunda v3, kadraj kenarindaki park
# halindeki arac/motorlu kurye/bina cephesini "tasan cop kutusu" saniyordu.
# Videonun ilk 2500 karesinden 500 "zor negatif" kare (yatay aynali) egitime
# eklendi (v4); kare>2500 HIC egitimde kullanilmadi (adil held-out). Sonuc:
# yanlis tespit 13 -> 3 kare (%77 azalma); cop1_ dogrulama recall'u ise
# %88,9 -> %74,1 (24/27 -> 20/27) dustu. Ek pozitif veri toplaninca recall
# geri kazanilacak; v3 best_cop_v3_arsiv.pt olarak saklandi.
#
# Madde 10 (2026-10-01): TasanCopKutusu TAMAMEN PROJEDEN CIKARILDI. Neden:
# kaynak veri setlerinin (cop1_/cop2_/cop3_/gbs_) buyuk kismi gercek Turkiye
# sokak/dashcam goruntusuyle gorsel olarak alakasizdi (gbs_'de Cince
# tabelalar, cop3_ aslinda elektrik/altyapi kutulari, digerleri rastgele
# acidan cekilmis plastik kucuk kutular) - bkz. rapor Bolum 12. Bu domain
# uyumsuzlugu, modelin gercek videoda farkli nesneleri (okul tasiti tabelasi
# gibi) surekli yanlislikla "cop kutusu" sanmasinin kok nedeniydi. Yerine
# projenin asil ihtiyaci olan KASIS (hiz kesici bozuklugu) ve YAYA GECIDI
# CIZGI BOZUKLUGU tespiti eklendi (bkz. asagidaki Kasis girisi, KASIS_SINIF_
# ADLARI ve gecit_analiz.py). best_cop.pt dosyasi models/ klasorunde arsiv
# olarak birakildi ama artik yuklenmiyor.
MODEL_PATHS = {
    "Cukur": os.path.join(BASE_DIR, "models", "best_cukur_seg.pt"),
    "HasarliTabela": os.path.join(BASE_DIR, "models", "best_tabela.pt"),
    "Kasis": os.path.join(BASE_DIR, "models", "best_kasis.pt"),
}

# Kasis modeli COK SINIFLI: YOLO'nun kendi ic sinif id'si (box.cls), bu
# listenin indeksi uzerinden GERCEK ariza turu adina cevrilir (bkz. main.py
# bir_tespiti_isle). Cukur/HasarliTabela tek sinifli oldugu icin listeleri
# tek elemanli - ayni mekanizmayi kullanirlar, davranislari degismez.
#
# "YayaGecidi" HAM (condition-agnostic) sinif adidir: kullandigimiz CDSet
# veri setinde etiketler SADECE "burada bir yaya gecidi var/yok" bilgisini
# tasir, bozuk/saglam ayrimi YOK (bkz. rapor Bolum 12 arastirmasi). Bu yuzden
# YOLO modeli sadece KONUMU bulur; BOZUKLUK DURUMU asagidaki klasik goruntu
# islemeyle (gecit_analiz.py) tespit SONRASI belirlenir - main.py bu ham
# etiketi, SADECE cok dusuk skorlu (asiri asinmis) durumlarda "GecitKullanilamaz"
# olarak YENIDEN ETIKETLER; digerlerinde (saglam ya da orta derecede asinmis)
# HIC RAPORLANMAZ (bkz. GECIT_KULLANILAMAZ_ESIK - "cok temkinli ol" ilkesi).
MODEL_SINIF_ADLARI = {
    "Cukur": ["Cukur"],
    "HasarliTabela": ["HasarliTabela"],
    "Kasis": ["KasisBozuk", "YayaGecidi"],
}

# gecit_analiz.cizgi_gorunurlugu_skoru() 0 (beyaz boya yok) - 1 (yogun beyaz
# boya) arasi bir skor dondurur (Otsu esiklemeyle beyaz piksel orani - bkz.
# gecit_analiz.py v2, 2026-10-01). TASARIM ILKESI (kullanicinin acik talebi):
# "cok temkinli olsun, ama asiri bozuk olani yakalasin" - yanlis NEGATIF
# (gercek bozuk bir gecidi kacirmak) kabul edilebilir, yanlis POZITIF (saglam
# gecidi bozuk sanmak) KABUL EDILEMEZ. Bu yuzden TEK ve COK DUSUK bir esik
# kullanilir, orta-seviye "GecitBozuk" durumu KALDIRILDI (basitlik + guvenlik
# icin): skor bu esigin USTUNDEYSE hic raporlanmaz (saglam/belirsiz -> ariza
# degil); esigin ALTINDAYSA "GecitKullanilamaz" olarak bildirilir. Esik,
# BagcılarYol'daki 8 GERCEK (hepsi farkli derecede asinmis ama hicbiri
# "asiri bozuk" olmayan) yaya gecidi orneginin skor araligina (0,30-0,66)
# gore, bunlarin HICBIRINI yanlislikla isaretlemeyecek kadar dusuk secildi.
GECIT_KULLANILAMAZ_ESIK = 0.10

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
# main.py, model.predict() cagrisinda MODEL_PREDICT_CONF (dusuk, genis bir
# aday havuzu icin) kullanir; GERCEK karar asagidaki MIN_GUVEN_ESIKLERI'nden
# (sinif bazli, varsayilan CONFIDENCE_THRESHOLD) gecer. Boylece her sinif
# icin ayri bir guven esigi uygulanabilir.
MODEL_PREDICT_CONF = 0.3
CONFIDENCE_THRESHOLD = 0.5  # Varsayilan - bu esigin altindaki tespitler yok sayilir

# KasisBozuk icin DAHA YUKSEK bir esik: BagcılarYol testinde parke tasi/islak
# zemin gibi seyleri 0,5-0,73 guvenle kasis sandigi tespit edildi (bkz. rapor
# Bolum 12.9.2). "Cok temkinli ol, asiri bozuk olani yakala" ilkesi geregi
# esik 0,8'e cikarildi - zor-negatif veri eklemesiyle BIRLIKTE calisir, tek
# basina yeterli degildir (bazi yanlis pozitifler 0,73 gibi yuksek guvenle
# gelmisti).
MIN_GUVEN_ESIKLERI = {
    "KasisBozuk": 0.8,
}

# Yanlis pozitifleri gecmek icin basit bir geometrik kural: her ariza turunun
# TIPIK en/boy orani farklidir. Gercek ariza_turu adi (MODEL_SINIF_ADLARI'ndan
# cozulmus) bu sozlukte varsa, kutunun genislik/yukseklik orani esigin
# ALTINDAYSA tespit atlanir (muhtemelen bu sinif degil).
#   Cukur (0.7): havadan/on kameradan bakildiginda enlemesine/kareye yakin
#     gorunur; yuruyen insan/motosikletli dikine, uzun bir siluet olusturur.
#   KasisBozuk/YayaGecidi (1.8): kasis ve yaya gecidi yolun GENISLIGINE
#     yayilan, cok genis-kisa yatay seritlerdir - bu yuksek esik, kompakt/
#     yuvarlaga yakin cukur/logar seklindeki nesnelerle KARISMASINI engeller
#     (bkz. rapor Bolum 12 - kasis/gecit eklenirken istenen gorsel ayrim).
#     YayaGecidi burada HAM sinif adidir (bozukluk durumu belirlenmeden
#     ONCE uygulanir - bkz. MODEL_SINIF_ADLARI aciklamasi).
# HasarliTabela bu sozlukte YOK: levha dogal olarak dikine/uzun bir kutu
# olabilir (levha dik durur), bu filtreyi uygulamak gercek tespitleri elerdi.
MIN_EN_BOY_ORANLARI = {
    "Cukur": 0.7,
    "KasisBozuk": 1.8,
    "YayaGecidi": 1.8,
}

# Bayrampasa sokak videosunda logar kapaklarinin cukur sanildigi bildirildi.
# DENENDI VE REDDEDILDI: "daire benzerligi" (4*pi*Alan/Cevre^2) ile logar
# kapaklarini geometrik olarak eleme fikri. Bayrampasa'daki dogrulanmis logar
# tespitlerinde bu deger 0.44-0.70 arasinda olcduldu (dusuk kamera acisindan
# bakildiginda logar bir elipse/yamuga donusuyor, mukemmel daire gorunmuyor).
# Ayni metrigi kendi egitim setimizdeki 356 GERCEK cukur maskesinde olctugumuzde
# medyan 0.71 cikti - yani gercek cukurlar logar tespitlerinden bile daha
# "daire benzeri" olabiliyor, iki grup tamamen ortusuyor. Bu yuzden geometrik
# filtre ISE YARAMIYOR ve eklenmedi. Cozum veri tabanli olmali: logar
# goruntuleri egitime ETIKETSIZ NEGATIF olarak eklenmeli (set2_universe'deki
# Manhole sinifi zaten bu sekilde kullaniliyor - bkz. merge_dataset_separate.py
# SET2_SRC, sadece Pothole=0 tutulup Manhole-only goruntuler negatif kaliyor;
# ayrica Bayrampasa'dan hard-negatif kare cikarimi planlaniyor).

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
