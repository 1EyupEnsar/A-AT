"""
DENENDI VE REDDEDILDI: 3 uzman modeli thread havuzuyla GERCEK paralel
calistirma denemesi.

Hipotez: PyTorch/CUDA cagrilari GIL'i cozdugu icin, 3 model.predict()
cagrisini bir ThreadPoolExecutor ile ayni anda baslatmak, kare basina
toplam sureyi 3 modelin TOPLAMINDAN, en yavas olanina yaklastirabilir.

Gercek olcum (RTX 4050 laptop GPU, ayni test videosu, 900 kare):
    Sirali (mevcut yontem): 40.6 ms/kare  (24.6 FPS)
    Paralel (bu modul):     68.8 ms/kare  (14.5 FPS)  -> ~%70 DAHA YAVAS

Nedeni: GPU tek, paylasilan bir kaynak. Yolov8n gibi kucuk modellerin
hesaplama suresi zaten kisa oldugu icin, thread'ler arasi GIL cekismesi
ve CUDA baglam (context) gecis maliyeti, hesaplamadan tasarruf edilen
sureden daha fazla. Yani bu donanim/model boyutu kombinasyonunda
"paralellik" kazandirmiyor, kaybettiriyor.

Bu dosya, main.py/web_video_test.py/test_video.py tarafindan ARTIK
KULLANILMIYOR (hepsi sirali yonteme geri donduruldu) - sadece bu
denemenin kaydini tutmak ve ayni hatanin tekrar denenmesini onlemek
icin projede saklanmaktadir (bkz. rapor - "gercek paralel calistirma"
gelecek calisma maddesi bu bulguyla birlikte guncellendi).
"""
from concurrent.futures import ThreadPoolExecutor


def paralel_tahmin_et(frame, modeller, conf_esigi):
    def tek_model_calistir(item):
        ariza_turu, model = item
        sonuc = model.predict(frame, conf=conf_esigi, verbose=False)[0]
        return ariza_turu, sonuc

    with ThreadPoolExecutor(max_workers=len(modeller)) as havuz:
        return list(havuz.map(tek_model_calistir, modeller))
