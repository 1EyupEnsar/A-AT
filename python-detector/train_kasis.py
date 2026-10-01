"""Kasis (hiz kesici + yaya gecidi) modelini TEK BASINA egitir (bkz.
merge_dataset_separate.py build_kasis() - nc=2: KasisBozuk, YayaGecidi).
Cukur/Tabela bu oturumda kasitli olarak DOKUNULMADI; train_separate.py'nin
3-model dongusu yerine bu izole script kullanilir.

Kesinti olursa (bkz. rapor - gece boyunca sik yasandi) AYNI komutu tekrar
calistir: runs/detect/uzman_kasis klasoru VARSA otomatik resume=True ile
devam eder."""
import os

from ultralytics import YOLO

EPOCHS = 80
IMG_SIZE = 640
# 2026-10-01: Egitim boyunca tekrar tekrar "sistem bellegi kritik seviyede
# dusuk" nedeniyle durduruldu (8 worker + batch=16 + mosaic, 15GB RAM'li
# makinede CPU tarafinda cok bellek aciydi). BATCH_SIZE 16->8, WORKERS 8->2
# dusuruldu - egitim biraz yavaslar ama kesinti sikligi azalir.
BATCH_SIZE = 8
WORKERS = 2
BASE_MODEL = "yolov8n.pt"
PATIENCE = 25
NAME = "uzman_kasis"

RUN_DIR = os.path.join("runs", "detect", NAME)
LAST_PT = os.path.join(RUN_DIR, "weights", "last.pt")


def main():
    if os.path.exists(LAST_PT):
        print(f"[RESUME] {LAST_PT} bulundu, egitime kaldigi yerden devam ediliyor.")
        model = YOLO(LAST_PT)
        model.train(resume=True, batch=BATCH_SIZE, workers=WORKERS)
    else:
        print(f"\n===== EGITIM BASLIYOR: {NAME} (dataset/model_kasis/data.yaml) =====\n")
        model = YOLO(BASE_MODEL)
        model.train(
            data="dataset/model_kasis/data.yaml",
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            workers=WORKERS,
            patience=PATIENCE,
            name=NAME,
        )
    print(f"\n===== EGITIM BITTI: {NAME} =====\n")


if __name__ == "__main__":
    main()
