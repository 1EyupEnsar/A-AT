# AŞAT — Akıllı Şehir Altyapı Arıza Tespit ve Yönetim Sistemi

TÜBİTAK 2209-A kapsamında geliştirilen, YOLOv8 ile görüntü işleme destekli
altyapı arıza tespit ve yönetim sistemi. Belediye araçlarına (örn. çöp
kamyonetleri) takılacak bir kamera ile normal rota sırasında 3 arıza türünü
otomatik tespit eder:

1. **Yol Çukuru** (Pothole) — küçük/orta/büyük kaba şiddet tahminiyle birlikte
2. **Taşan Çöp Kutusu** (Overflowing Trash Bin)
3. **Hasarlı / Yıkılmış Trafik Levhası** (Damaged Traffic Sign)

Her tür için **ayrı, kendi konusunda uzmanlaşmış bir YOLOv8 modeli** kullanılır
(neden tek bir 3-sınıflı model yerine 3 ayrı model kullanıldığı, projenin
raporundaki "paylaşılan kapasite rekabeti" bulgusuyla açıklanır).

## Mimari

```
AkilliSehirArizaTespit/
├── python-detector/     Görüntü işleme servisi (Python + YOLOv8 + OpenCV)
│   ├── config.py            Merkezi ayarlar (model yolları, eşikler, GPS)
│   ├── main.py              Ana tespit döngüsü (video → 3 model → API)
│   ├── test_video.py        Claude/Backend olmadan bağımsız video testi
│   ├── merge_dataset_separate.py   3 ayrı tek-sınıflı veri seti oluşturur
│   ├── train_separate.py    3 uzman modeli sırayla eğitir
│   └── models/              Eğitilmiş ağırlıklar (best_cukur/tabela/cop.pt)
└── backend/
    └── ArizaTakipSistemi/   ASP.NET Core MVC (API + Web Paneli + Admin Paneli, SQLite ile)
```

Akış: `main.py` videodan/kameradan kare okur → kareyi **3 uzman modelden**
sırayla geçirir (her model sadece kendi türünü arar) → geometrik filtre
(insan/araç elemek için, sadece Çukur) → soğuma süresi + **konum tabanlı
tekrar-önleme** (aynı arıza 20m yakınında zaten "Bekliyor" durumunda bir
kayıt varsa atlanır) → kareyi kırpıp `wwwroot/uploads/`'a kaydeder → çukurlar
için kaba bir Küçük/Orta/Büyük şiddet hesaplar → `POST /api/ariza` ile
ASP.NET Core'a JSON gönderir → kayıt SQLite'a (`ariza.db`) yazılır →
Dashboard bu kayıtları listeler/haritalar/filtreler; şifreyle korunan
**Admin Paneli**'nden personel kayıtları manuel ekleyip/düzenleyip/silebilir.

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

Eğitilmiş 3 uzman model (`models/best_cukur.pt`, `best_tabela.pt`,
`best_cop.pt`) repoya dahildir, elle bir şey eklemene gerek yok. Kendi test
videonu `python-detector/videos/` altına koyup `config.py` → `VIDEO_PATH`'i
güncelle (ya da doğrudan `test_video.py video_yolu.mp4` ile bağımsız test et).

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

## Model Eğitimi — 3 Ayrı Uzman Model

Her arıza türü kendi tek-sınıflı veri setinde, diğerlerinden habersiz olarak
eğitilir (bkz. proje raporu Bölüm 6.5 — bunun nedeni, tek bir küçük modelde
(yolov8n) 3 sınıfı birleştirmenin "paylaşılan kapasite rekabetine" yol
açması). Ayrıca, birbirine görsel olarak benzeyen sınıflar (çukur ↔ dolu
çöp kutusu) arasında çapraz karışıklığı önlemek için **çapraz-negatif**
örnekler enjekte edilir (bkz. Bölüm 6.6).

1. Roboflow Universe'den her sınıf için ayrı, konuyla doğrudan ilgili
   (çukur / hasarlı trafik levhası / taşan çöp kutusu) veri setleri
   toplanır ve `python-detector/dataset/<kaynak>_universe/` altına
   YOLOv8 formatında indirilir.
2. `merge_dataset_separate.py` çalıştırılır — bu, `dataset/model_cukur/`,
   `dataset/model_tabela/`, `dataset/model_cop/` adlı 3 ayrı tek-sınıflı
   veri seti üretir (her birine diğer sınıfların görüntülerini etiketsiz
   "negatif" örnek olarak da ekler).
3. `train_separate.py` çalıştırılır — 3 modeli sırayla (`yolov8n`, 640px,
   80 epoch) eğitir.
4. Sonuçlar `runs/detect/uzman_*/weights/best.pt` altında oluşur;
   `models/best_cukur.pt`, `best_tabela.pt`, `best_cop.pt` olarak kopyalanır.

Tam sayısal sonuçlar ve karşılaşılan sorunlar (veri sulandırma, çözünürlük
denemesi, kapasite rekabeti, çapraz karışıklık) için masaüstündeki
**AŞAT Proje Raporu.docx** ve **AŞAT SPICE Raporu.docx** dosyalarına bakınız.

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
