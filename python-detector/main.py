"""
Akilli Sehir Altyapi Ariza Tespit Sistemi - Goruntu Isleme Servisi
====================================================================

Bir video akisini (dosya veya kamera) okur, YOLOv8 ile altyapi
arizalarindan (yol cukuru, hasarli trafik levhasi, kasis bozuklugu,
yaya gecidi cizgi bozuklugu/kullanilmazligi) birini tespit ettiginde:

  1) Tespit edilen bolgeyi kareden kirpip .jpg olarak kaydeder,
  2) Backend'e (ASP.NET Core) JSON POST istegi atarak kaydi olusturur.

Aynı arizanin ust uste her karede (saniyede ~30 kez) tekrar tekrar
kaydedilmesini onlemek icin sinif bazli bir "cooldown" (soguma suresi)
mantigi kullanilir.

Calistirmak icin:
    pip install -r requirements.txt
    python main.py
"""
import math
import os
import time
import uuid
from datetime import datetime, timezone

import cv2
import requests
from ultralytics import YOLO

import config
import gecit_analiz


class KonumSimulatoru:
    """Kamyonetin bir rota boyunca ilerledigini simule ederek her tespit
    icin bir enlem/boylam uretir. Gercek donanimda bu sinif, GPS
    modulunden (orn. seri port uzerinden NMEA verisi okuyarak) anlik
    konumu donduren bir siniffonksiyonla degistirilmelidir; geri kalan
    kod (API'ye enlem/boylam gonderme) aynen calismaya devam eder."""

    def __init__(self, baslangic_enlem, baslangic_boylam, adim_enlem, adim_boylam):
        self.enlem = baslangic_enlem
        self.boylam = baslangic_boylam
        self.adim_enlem = adim_enlem
        self.adim_boylam = adim_boylam

    def su_anki_konum(self):
        self.enlem += self.adim_enlem
        self.boylam += self.adim_boylam
        return round(self.enlem, 6), round(self.boylam, 6)


class CooldownTakipci:
    """Her arizatipi icin son gonderim zamanini tutar."""

    def __init__(self, cooldown_seconds: float):
        self.cooldown_seconds = cooldown_seconds
        self._son_gonderim = {}

    def gonderilebilir_mi(self, ariza_turu: str) -> bool:
        simdi = time.monotonic()
        son = self._son_gonderim.get(ariza_turu, 0.0)
        return (simdi - son) >= self.cooldown_seconds

    def isaretle(self, ariza_turu: str) -> None:
        self._son_gonderim[ariza_turu] = time.monotonic()


def klasorleri_hazirla():
    os.makedirs(config.UPLOAD_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(config.VIDEO_PATH), exist_ok=True)
    for model_path in config.MODEL_PATHS.values():
        os.makedirs(os.path.dirname(model_path), exist_ok=True)


def kucult_onizleme(frame, max_genislik=1280):
    """Onizleme penceresinin ekrana sigmasi icin yuksek cozunurluklu
    (orn. 4K) kareleri oranini koruyarak kucultur; tespit/kayit
    islemleri hala orijinal boyuttaki `frame` uzerinde yapilir."""
    h, w = frame.shape[:2]
    if w <= max_genislik:
        return frame
    olcek = max_genislik / w
    return cv2.resize(frame, (max_genislik, int(h * olcek)))


def kareyi_kirp(frame, x1, y1, x2, y2, pay=10):
    """Kutuyu kucuk bir payla birlikte kirpar, kare disina tasmasini engeller."""
    h, w = frame.shape[:2]
    x1 = max(0, int(x1) - pay)
    y1 = max(0, int(y1) - pay)
    x2 = min(w, int(x2) + pay)
    y2 = min(h, int(y2) + pay)
    return frame[y1:y2, x1:x2]


def goruntuyu_kaydet(kirpik, ariza_turu: str) -> str:
    """Kirpilan goruntuyu diske kaydeder, backend'e gonderilecek relatif yolu dondurur."""
    dosya_adi = f"{ariza_turu}_{uuid.uuid4().hex[:10]}.jpg"
    tam_yol = os.path.join(config.UPLOAD_DIR, dosya_adi)
    cv2.imwrite(tam_yol, kirpik)
    # Backend bu klasoru "/uploads" olarak statik servis ediyor.
    return f"/uploads/{dosya_adi}"


def siddet_hesapla(
    ariza_turu: str, genislik: float, yukseklik: float,
    kare_genislik: int, kare_yukseklik: int,
) -> str | None:
    """Tespit kutusunun kare alanina oranina gore kaba bir 'Kucuk/Orta/Buyuk'
    siddet tahmini uretir. Sadece config.SIDDET_UYGULANACAK_TURLER icindeki
    turler icin anlamlidir (bkz. config.py aciklamasi); digerlerinde None
    doner (Kaggle'daki segmentasyon-alani fikrinden esinlenilen, kutu
    tabanli basitlestirilmis versiyon)."""
    if ariza_turu not in config.SIDDET_UYGULANACAK_TURLER:
        return None
    kare_alani = kare_genislik * kare_yukseklik
    if kare_alani <= 0:
        return None
    oran = (genislik * yukseklik) / kare_alani
    if oran < config.SIDDET_ESIK_KUCUK_ORTA:
        return "Kucuk"
    if oran < config.SIDDET_ESIK_ORTA_BUYUK:
        return "Orta"
    return "Buyuk"


def poligon_alani(xy) -> float:
    """Bir maske poligonunun (piksel koordinatlarinda, Nx2 dizi) alanini
    Shoelace formuluyle hesaplar."""
    x = xy[:, 0]
    y = xy[:, 1]
    return 0.5 * abs(
        (x * (y[list(range(1, len(y))) + [0]] - y[list(range(-1, len(y) - 1))])).sum()
    )


def siddet_hesapla_maskeden(
    ariza_turu: str, maske_xy, kare_genislik: int, kare_yukseklik: int,
) -> str | None:
    """siddet_hesapla ile AYNI esikleri kullanir, ama kutu alani yerine
    segmentasyon modelinin urettigi GERCEK maske poligon alanini kullanir
    (bkz. Madde 8 - Kaggle'daki fikrin tam uygulamasi: kutu yerine gercek
    piksel segmentasyonu). Sadece segmentasyon maskesi mevcutsa (yani Cukur
    modeli yolov8n-SEG ise) cagrilir; digerlerinde siddet_hesapla (kutu
    tabanli) kullanilmaya devam eder."""
    if ariza_turu not in config.SIDDET_UYGULANACAK_TURLER:
        return None
    kare_alani = kare_genislik * kare_yukseklik
    if kare_alani <= 0 or maske_xy is None or len(maske_xy) < 3:
        return None
    oran = poligon_alani(maske_xy) / kare_alani
    if oran < config.SIDDET_ESIK_KUCUK_ORTA:
        return "Kucuk"
    if oran < config.SIDDET_ESIK_ORTA_BUYUK:
        return "Orta"
    return "Buyuk"


def iki_konum_mesafesi_metre(enlem1, boylam1, enlem2, boylam2):
    """Kucuk mesafeler (sehir ici) icin duzlem yaklasimiyla iki enlem/boylam
    arasindaki mesafeyi metre cinsinden hesaplar (haversine gerektirmeyecek
    kadar kucuk mesafeler icin yeterince dogru)."""
    derece_metre = 111_320  # 1 derece enlem ~ 111.32 km
    dy = (enlem2 - enlem1) * derece_metre
    dx = (boylam2 - boylam1) * derece_metre * math.cos(math.radians(enlem1))
    return math.hypot(dx, dy)


def yakinda_acik_kayit_var_mi(ariza_turu: str, enlem: float, boylam: float) -> bool:
    """Backend'de ayni turden, hala 'Bekliyor' durumunda ve bu koordinata
    config.DUPLICATE_MESAFE_METRE'den yakin bir kayit olup olmadigini sorar.
    Boylece ayni fiziksel ariza (video tekrar izlendiginde ya da ayni yoldan
    tekrar gecildiginde) birden fazla kez kaydedilmez. API'ye ulasilamazsa
    (agir hata degil) kontrolu atlayip False doner, sistem calismaya devam eder."""
    try:
        response = requests.get(
            config.API_ENDPOINT,
            params={"tur": ariza_turu, "durum": "Bekliyor"},
            timeout=config.REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        for kayit in response.json():
            mevcut_enlem = kayit.get("enlem")
            mevcut_boylam = kayit.get("boylam")
            if mevcut_enlem is None or mevcut_boylam is None:
                continue
            mesafe = iki_konum_mesafesi_metre(enlem, boylam, mevcut_enlem, mevcut_boylam)
            if mesafe <= config.DUPLICATE_MESAFE_METRE:
                return True
        return False
    except requests.exceptions.RequestException as exc:
        print(f"[API] Yakin kayit kontrolu basarisiz (atlaniyor): {exc}")
        return False


def api_ye_gonder(
    ariza_turu: str,
    fotograf_yolu: str,
    guven_skoru: float,
    enlem: float | None = None,
    boylam: float | None = None,
    siddet: str | None = None,
) -> bool:
    payload = {
        "arizaTuru": ariza_turu,
        "tespitZamani": datetime.now(timezone.utc).isoformat(),
        "fotografYolu": fotograf_yolu,
        "guvenSkoru": round(guven_skoru, 4),
        "enlem": enlem,
        "boylam": boylam,
        "siddet": siddet,
    }
    try:
        response = requests.post(
            config.API_ENDPOINT, json=payload, timeout=config.REQUEST_TIMEOUT
        )
        if response.status_code in (200, 201):
            konum_bilgisi = f", konum=({enlem}, {boylam})" if enlem is not None else ""
            siddet_bilgisi = f", siddet={siddet}" if siddet is not None else ""
            print(f"[API] Kaydedildi -> {ariza_turu} (guven={guven_skoru:.2f}{konum_bilgisi}{siddet_bilgisi})")
            return True
        print(f"[API] Beklenmeyen durum kodu {response.status_code}: {response.text}")
        return False
    except requests.exceptions.RequestException as exc:
        print(f"[API] Istek basarisiz: {exc}")
        return False


def bir_tespiti_isle(frame, model_adi, kutu, cooldown, konum, maske_xy=None):
    """Tek bir modelin dondurdugu tek bir kutuyu isler: filtre, onizleme,
    soguma, kirpma, siddet, API gonderimi. Ucu uzman model icin de ayni
    mantik kullanilir, sadece girdi kutusu farkli modelden gelir.

    model_adi: config.MODEL_PATHS'teki anahtar (orn. "Kasis"). Modelin
    KENDI sinif id'si (kutu.cls), config.MODEL_SINIF_ADLARI[model_adi]
    listesi uzerinden GERCEK ariza turu adina (orn. "KasisBozuk" ya da
    "GecitBozuk") cevrilir - Cukur/HasarliTabela tek sinifli oldugu icin
    bu onlarda hep ayni ismi dondurur, davranis degismez.

    maske_xy: Cukur modeli segmentasyon modeliyse (yolov8n-seg), o tespitin
    GERCEK piksel maskesinin poligon koordinatlari (Madde 8); verilmezse
    (None) kutu-alani tabanli eski yonteme (siddet_hesapla) geri dusulur."""
    sinif_id = int(kutu.cls[0])
    ariza_turu = config.MODEL_SINIF_ADLARI[model_adi][sinif_id]

    guven = float(kutu.conf[0])
    guven_esigi = config.MIN_GUVEN_ESIKLERI.get(ariza_turu, config.CONFIDENCE_THRESHOLD)
    if guven < guven_esigi:
        # Bu sinif icin genel esikten (MODEL_PREDICT_CONF) gecmis ama sinif
        # bazli, daha siki esigi gecemedi (bkz. config.py MIN_GUVEN_ESIKLERI).
        return

    x1, y1, x2, y2 = kutu.xyxy[0].tolist()
    genislik = x2 - x1
    yukseklik = y2 - y1

    esik = config.MIN_EN_BOY_ORANLARI.get(ariza_turu)
    if esik is not None:
        if yukseklik <= 0 or (genislik / yukseklik) < esik:
            # Beklenen en/boy oranina uymuyor -- Cukur icin dikine/uzun bir
            # insan/arac siluetiyse, Kasis/GecitBozuk icin kompakt/yuvarlaga
            # yakin bir cukur/logar siluetiyse atla (bkz. config.py notu).
            return

    if ariza_turu == "YayaGecidi":
        # YOLO sadece KONUMU buldu (bkz. config.py notu); BOZUKLUK DURUMU
        # klasik beyaz-piksel-orani analiziyle belirlenir. "Cok temkinli ol"
        # ilkesi geregi SADECE skor cok dusukse (asiri asinmis/silinmis)
        # raporlanir; saglam ya da orta derecede asinmis gecitler (yanlis
        # pozitif riskini dusurmek icin) hic raporlanmaz.
        kirpik_analiz = kareyi_kirp(frame, x1, y1, x2, y2, pay=10)
        skor = gecit_analiz.cizgi_gorunurlugu_skoru(kirpik_analiz)
        if skor >= config.GECIT_KULLANILAMAZ_ESIK:
            return  # Saglam/belirsiz -- ariza degil, raporlanmaz.
        ariza_turu = "GecitKullanilamaz"

    if config.SHOW_PREVIEW:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(
            frame, f"{ariza_turu} {guven:.2f}", (int(x1), max(0, int(y1) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2,
        )

    if not cooldown.gonderilebilir_mi(ariza_turu):
        return  # Bu ariza turu icin soguma suresi henuz gecmedi

    # Bu sinif icin soguma suresini burada isaretliyoruz (basarili olsun ya
    # da olmasin) ki asagidaki "yakinda acik kayit var mi" kontrolu her
    # karede degil, en fazla COOLDOWN_SECONDS'te bir backend'e sorulsun.
    cooldown.isaretle(ariza_turu)

    enlem, boylam = konum.su_anki_konum()
    if yakinda_acik_kayit_var_mi(ariza_turu, enlem, boylam):
        # Ayni fiziksel ariza zaten "Bekliyor" durumunda kayitli -- video
        # tekrar izlense ya da ayni yoldan tekrar gecilse bile ayni arizayi
        # tekrar tekrar kaydetmeyelim.
        return

    kirpik = kareyi_kirp(frame, x1, y1, x2, y2)
    if kirpik.size == 0:
        return

    kare_yukseklik, kare_genislik = frame.shape[:2]
    if maske_xy is not None:
        siddet = siddet_hesapla_maskeden(ariza_turu, maske_xy, kare_genislik, kare_yukseklik)
    else:
        siddet = siddet_hesapla(ariza_turu, genislik, yukseklik, kare_genislik, kare_yukseklik)

    fotograf_yolu = goruntuyu_kaydet(kirpik, ariza_turu)
    api_ye_gonder(ariza_turu, fotograf_yolu, guven, enlem, boylam, siddet)


def main():
    klasorleri_hazirla()

    modeller = []
    for ariza_turu, model_path in config.MODEL_PATHS.items():
        print(f"[MODEL] Yukleniyor ({ariza_turu}): {model_path}")
        modeller.append((ariza_turu, YOLO(model_path)))

    print(f"[VIDEO] Aciliyor: {config.VIDEO_PATH}")
    capture = cv2.VideoCapture(config.VIDEO_PATH)
    if not capture.isOpened():
        raise RuntimeError(
            f"Video acilamadi: {config.VIDEO_PATH}. "
            "config.py icindeki VIDEO_PATH degerini kontrol et."
        )

    cooldown = CooldownTakipci(config.COOLDOWN_SECONDS)
    konum = KonumSimulatoru(
        config.GPS_BASLANGIC_ENLEM,
        config.GPS_BASLANGIC_BOYLAM,
        config.GPS_ADIM_ENLEM,
        config.GPS_ADIM_BOYLAM,
    )

    try:
        while True:
            basarili, frame = capture.read()
            if not basarili:
                print("[VIDEO] Video sona erdi ya da kare okunamadi.")
                break

            # Her uzman model kendi turunu ariyor; her biri kareyi ayri
            # ayri, birbirinden habersiz sekilde degerlendiriyor.
            #
            # NOT: Bu dongu bilincli olarak SIRALIDIR. Thread havuzuyla
            # GERCEK paralel calistirma denendi (bkz. coklu_model.py, artik
            # kullanilmiyor) ve olculdu: paralel yontem kare basina 40,6ms'den
            # 68,8ms'ye CIKARDI (~%70 YAVASLAMA), hizlandirmadi. Nedeni: GPU
            # tek bir paylasilan kaynak; 3 kucuk modelin (yolov8n) thread'ler
            # arasinda GIL + CUDA baglam gecisi maliyeti, zaten hizli olan
            # hesaplamadan daha pahaliya geliyor. Bu yuzden sirali yontem
            # korunmustur (bkz. rapor - "denendi, gercek olcumle reddedildi").
            for model_adi, model in modeller:
                sonuclar = model.predict(
                    frame, conf=config.MODEL_PREDICT_CONF, verbose=False
                )[0]
                # Segmentasyon modeli (yolov8n-seg, sadece Cukur icin -
                # Madde 8) hem kutu hem maske dondurur; kutu tabanli
                # modellerde sonuclar.masks None olur.
                maskeler = sonuclar.masks
                for i, kutu in enumerate(sonuclar.boxes):
                    maske_xy = maskeler.xy[i] if maskeler is not None else None
                    bir_tespiti_isle(frame, model_adi, kutu, cooldown, konum, maske_xy)

            if config.SHOW_PREVIEW:
                onizleme = kucult_onizleme(frame, max_genislik=1280)
                cv2.imshow("Akilli Sehir - Ariza Tespiti (q: cikis)", onizleme)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()
        print("[SISTEM] Servis durduruldu.")


if __name__ == "__main__":
    main()
