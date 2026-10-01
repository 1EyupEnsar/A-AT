# AŞAT — Akıllı Şehir Altyapı Arıza Tespit ve Yönetim Sistemi

TÜBİTAK 2209-A kapsamında geliştirilen, YOLOv8 ile görüntü işleme destekli
altyapı arıza tespit ve yönetim sistemi. Belediye araçlarına (örn. çöp
kamyonetleri) takılacak bir kamera ile normal rota sırasında altyapı
arızalarını otomatik tespit eder:

**Üretimde (production):**
1. **Yol Çukuru** (Pothole) — segmentasyon modeli (gerçek piksel maskesi),
   küçük/orta/büyük kaba şiddet tahminiyle birlikte
2. **Hasarlı / Yıkılmış Trafik Levhası** (Damaged Traffic Sign)

**Geliştirme/test aşamasında (henüz üretime alınmadı):**
3. **Kasis Bozukluğu** (Speed Bump Damage)
4. **Yaya Geçidi Çizgi Bozukluğu/Kullanılamazlığı** (Crosswalk Line Wear)

> **Not (2026-10-01):** Önceki sürümde üçüncü sınıf olarak "Taşan Çöp
> Kutusu" vardı; kaynak veri setlerinin gerçek Türkiye sokak görüntüsüyle
> alakasız olduğu tespit edilince TAMAMEN KALDIRILDI, yerine Kasis/Yaya
> Geçidi eklendi (detaylar: proje raporu Bölüm 12.9). Kasis modeli ilk
> gerçek-dünya testinde yüksek yanlış pozitif oranı gösterdi (parke taşı
> döşeli yollarla karıştırıyordu); zor-negatif veri eklenip eşikler
> sıkılaştırılarak yeniden eğitiliyor, henüz `models/` altında değil.

Her tür için **ayrı, kendi konusunda uzmanlaşmış bir YOLOv8 modeli** kullanılır
(neden tek bir çok-sınıflı model yerine ayrı modeller kullanıldığı, projenin
raporundaki "paylaşılan kapasite rekabeti" bulgusuyla açıklanır). İstisna:
Kasis modeli KasisBozuk+YayaGecidi için tek, 2 sınıflı bir model (ikisi de
"yolun genişliğine yayılan yatay şerit" geometrisini paylaştığı için ayrı
modele gerek görülmedi).

## Mimari

```
AkilliSehirArizaTespit/
├── python-detector/     Görüntü işleme servisi (Python + YOLOv8 + OpenCV)
│   ├── config.py            Merkezi ayarlar (model yolları, esikler, GPS)
│   ├── main.py              Ana tespit döngüsü (video → modeller → API)
│   ├── gecit_analiz.py      Yaya geçidi çizgi-görünürlüğü analizi (Otsu)
│   ├── merge_dataset_seg.py       Çukur (segmentasyon) veri setini oluşturur
│   ├── merge_dataset_separate.py  Tabela + Kasis veri setlerini oluşturur
│   ├── train_cukur_seg.py   Çukur segmentasyon modelini eğitir
│   ├── train_kasis.py       Kasis/YayaGecidi modelini eğitir (resume destekli)
│   └── models/              Eğitilmiş ağırlıklar (best_cukur_seg/tabela.pt)
└── backend/
    └── ArizaTakipSistemi/   ASP.NET Core MVC (API + Web Paneli + Admin Paneli, SQLite ile)
```

Akış: `main.py` videodan/kameradan kare okur → kareyi **uzman modellerden**
sırayla geçirir (her model sadece kendi türünü arar) → sınıf bazlı geometrik
en/boy filtresi (Çukur: insan/araç elemek için; Kasis/YayaGecidi: ters
yönde, kompakt çukur/logar şekilleriyle karışmasın diye) → KasisBozuk için
ek, daha yüksek bir güven eşiği → YayaGecidi için çizgi-görünürlüğü analizi
(`gecit_analiz.py`: sağlam/belirsizse hiç raporlanmaz, sadece aşırı aşınmışsa
raporlanır) → soğuma süresi + **konum tabanlı tekrar-önleme** (aynı arıza
20m yakınında zaten "Bekliyor" durumunda bir kayıt varsa atlanır) → kareyi
kırpıp `wwwroot/uploads/`'a kaydeder → çukurlar için gerçek maske alanından
kaba bir Küçük/Orta/Büyük şiddet hesaplar → `POST /api/ariza` ile ASP.NET
Core'a JSON gönderir → kayıt SQLite'a (`ariza.db`) yazılır → Dashboard bu
kayıtları listeler/haritalar/filtreler; şifreyle korunan **Admin
Paneli**'nden personel kayıtları manuel ekleyip/düzenleyip/silebilir.

## Gereksinimler

- [.NET 8 SDK](https://dotnet.microsoft.com/download)
- Python 3.10+ (test edilen sürüm: 3.13)
- (Opsiyonel, GPU ile hızlı eğitim/çıkarım için) CUDA destekli PyTorch

## Çalıştırma Sırası (adım adım)

Sistemde **iki bağımsız süreç** var: backend (ASP.NET Core, sürekli açık
kalır) ve Python tespit servisi (backend çalışırken başlatılır). İki ayrı
terminal gerekir; backend önce ayakta olmalı çünkü Python ona istek atıyor.

### Adım 0 — Ön koşulları doğrula

```powershell
dotnet --version
python --version
```

### Adım 1 — Terminal #1: Backend'i başlat

```powershell
cd C:\Users\lasto\AkilliSehirArizaTespit\backend\ArizaTakipSistemi
dotnet restore
dotnet run --urls http://localhost:5080
```

Konsolda `Now listening on: http://localhost:5080` görünmeli. Tarayıcıda
`http://localhost:5080/Dashboard` panele, `http://localhost:5080/Admin`
(şifre: `appsettings.json` → `AdminAyarlari:Sifre`) yönetim paneline açılır.
**Bu terminali kapatma** — backend, Python çalışırken arka planda açık kalmalı.

### Adım 2 — Terminal #2: Python ortamını hazırla

```powershell
cd C:\Users\lasto\AkilliSehirArizaTespit\python-detector
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -c "import cv2, ultralytics, requests; print('Bagimliliklar OK')"
```

### Adım 3 — Modeller ve test videosu

Üretimdeki 2 model (`models/best_cukur_seg.pt`, `best_tabela.pt`) repoya
dahildir, elle bir şey eklemene gerek yok. `best_cop.pt` (eski, artık
kullanılmayan çöp kutusu modeli) sadece arşiv/referans amaçlı repoda kalmış,
`config.py` onu yüklemiyor. Kasis modeli (`best_kasis.pt`) henüz yok — hâlâ
eğitim/test aşamasında, `config.py`'deki `MODEL_PATHS["Kasis"]` yolu şu an
mevcut olmayan bir dosyayı gösteriyor; Kasis'i denemek istersen önce
`train_kasis.py` ile eğitip `runs/detect/uzman_kasis/weights/best.pt`'yi
`models/best_kasis.pt` olarak kopyalaman gerekir. Kendi test videonu
`python-detector/videos/` altına koyup `config.py` → `VIDEO_PATH`'i güncelle.

### Adım 4 — Python servisini çalıştır

```powershell
python main.py
```

Bir önizleme penceresi açılır; tespit edilen nesnelerin etrafına kutu ve
etiket çizilir. **`q`** ile düzgünce kapatılır. Arka planda (penceresiz)
çalıştırmak için `config.py` → `SHOW_PREVIEW = False`.

### Adım 5 — Sonucu Dashboard'da doğrula

Python konsolunda `[API] Kaydedildi -> ...` gördükten sonra Dashboard'u
yenile (F5); yeni tespit fotoğrafıyla birlikte tabloda görünmeli.

### Sık Karşılaşılan Hatalar

| Hata | Sebep / Çözüm |
|---|---|
| `[API] Istek basarisiz: ... Connection refused` | Backend (Adım 1) çalışmıyor. `config.py` → `API_BASE_URL`'i kontrol et. |
| `Video acilamadi: ...` | `config.py` → `VIDEO_PATH` yanlış ya da dosya yok. |
| Dashboard'da yeni tespit görünmüyor | Sayfayı yenile (F5) — panel otomatik yenilenmiyor. |
| Ondalıklı sayı alanı (Admin panelinde) hatalı kaydediliyor | Program.cs'de `InvariantCulture` ayarı olmalı — bu repoda zaten düzeltilmiş durumda. |

## Yönetim Paneli (Admin)

`/Admin` adresinden, `appsettings.json`'daki şifreyle giriş yapılır. Buradan:
kayıt ekleme (fotoğraf yükleme dahil), herhangi bir kaydın tüm alanlarını
düzenleme, kayıt silme yapılabilir — tamamen otomatik tespitlerin dışında,
personelin manuel müdahalesi içindir. Tüm formlar CSRF (antiforgery)
korumalıdır.

**Not:** `appsettings.json` içindeki `AdminAyarlari:Sifre` düz metin olarak
tutuluyor; repo private olduğu için kabul edilebilir, public'e alınacaksa
bu değerin bir ortam değişkenine taşınması önerilir.

## Model Eğitimi — Ayrı Uzman Modeller

Her arıza türü kendi veri setinde, diğerlerinden habersiz olarak eğitilir
(bkz. proje raporu Bölüm 6.5 — bunun nedeni, tek bir küçük modelde (yolov8n)
birden fazla sınıfı birleştirmenin "paylaşılan kapasite rekabetine" yol
açması). Birbirine görsel olarak benzeyen sınıflar arasında çapraz
karışıklığı önlemek için **çapraz-negatif** örnekler enjekte edilir (bkz.
Bölüm 6.6, 12.9).

### Çukur (segmentasyon)

1. Roboflow Universe + RDD2022 + Smartathon gibi kaynaklardan çukur
   görüntüleri toplanır (`dataset/<kaynak>_universe/`).
2. `merge_dataset_seg.py` çalıştırılır → `dataset/model_cukur/` (segmentasyon
   etiketli, tek sınıf) oluşur.
3. `train_cukur_seg.py` çalıştırılır (`yolov8n-seg`, 640px).
4. `runs/detect/cukur_segmentasyon/weights/best.pt` → `models/best_cukur_seg.pt`

### Hasarlı Tabela

`merge_dataset_separate.py` içindeki `TABELA_SOURCES` + `build_from_sources`
ile `dataset/model_tabela/` oluşturulur, `train_separate.py` ile eğitilir.

### Kasis + Yaya Geçidi (geliştirme aşamasında)

1. Roboflow "Humps/Bumps & Potholes Detection" (sadece Speed Bump sınıfı) +
   Zenodo CDSet (yaya geçidi konumu, CC BY 4.0) indirilir.
2. `merge_dataset_separate.py` içindeki `build_kasis()` çalıştırılır →
   `dataset/model_kasis/` (2 sınıflı: KasisBozuk=0, YayaGecidi=1) oluşur.
   Gerçek videodan çıkarılan zor-negatifler de (`kasis_negatif_cikar.py`)
   buraya eklenir.
3. `augment_dataset.py model_kasis <poz_çarpan> <neg_çarpan>` ile çoğaltılır.
4. `train_kasis.py` çalıştırılır (`yolov8n`, 640px; kesinti olursa aynı
   komutu tekrar çalıştırmak otomatik `resume=True` ile devam eder).
5. YayaGecidi'nin bozuk/sağlam DURUMU, YOLO'dan BAĞIMSIZ olarak
   `gecit_analiz.py` ile (Otsu eşiklemeli beyaz-piksel-oranı) belirlenir —
   YOLO sadece konumu bulur.
6. `runs/detect/uzman_kasis/weights/best.pt` → (testler başarılıysa)
   `models/best_kasis.pt`.

Tam sayısal sonuçlar, başarısız denemeler (TasanCopKutusu'nun neden
kaldırıldığı, Kasis'in ilk versiyonunun neden güvenilmez bulunduğu, geçit
analiz algoritmasındaki hata) için masaüstündeki **AŞAT Proje Raporu.docx**
dosyasına (özellikle Bölüm 12) bakınız — bu, projenin TEK güncel tutulan
kapsamlı kaynağıdır.

## Tekrar-Önleme (Duplicate Detection)

Aynı fiziksel arızanın (aynı video tekrar izlense ya da araç aynı yoldan
tekrar geçse bile) birden fazla kez kaydedilmesini önlemek için, yeni bir
tespit kaydedilmeden önce backend'e "bu türden, hâlâ Bekliyor durumunda, bu
koordinata 20 metreden yakın bir kayıt var mı?" diye sorulur (bkz.
`main.yakinda_acik_kayit_var_mi`). Sınıf bazlı 8 saniyelik cooldown
(`config.COOLDOWN_SECONDS`) buna ek, daha kısa vadeli bir hız sınırlayıcıdır.

## Notlar

- Veritabanı şeması değiştiğinde `ariza.db`'yi silip `dotnet run` ile
  yeniden oluşturabilirsin (`Database.EnsureCreated()` kullanılıyor, EF
  migration yok) — ama mevcut veriyi kaybetmemek için önce yedekle.
- `wwwroot/uploads/` klasörü Python'un yazdığı fotoğrafları barındırır.
- `python-detector/train.py`, `merge_dataset.py`, `build_test_video*.py`
  eski/yerine yenisi gelmiş scriptlerdir, repoya dahil edilmemiştir
  (bkz. `.gitignore`) ama referans için diskte kalabilir.
