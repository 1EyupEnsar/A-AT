"""
Tam hat: birlestir -> augmente et (sadece train) -> 3 uzman modeli egit.
Her asama isaret dosyasi birakir; bellek baskisiyla surec olurse ayni komut
tekrar calistirilinca kaldigi yerden (egitimde resume=True) devam eder.
Sira: cukur (oncelik) -> cop -> tabela.
"""
import os
import subprocess
import sys

import torch
from ultralytics import YOLO

BASE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
ISARET = os.path.join(BASE, "dataset", ".asama")
os.makedirs(ISARET, exist_ok=True)


def once(ad, komut):
    yol = os.path.join(ISARET, ad)
    if os.path.exists(yol):
        print(f"[atla] {ad}", flush=True)
        return
    print(f"[calistir] {ad}: {' '.join(komut)}", flush=True)
    subprocess.run(komut, check=True, cwd=BASE)
    open(yol, "w").close()


def egit(ad, data, epochs, patience):
    run = os.path.join(BASE, "runs", "detect", ad)
    last = os.path.join(run, "weights", "last.pt")
    if os.path.exists(last):
        ck = torch.load(last, map_location="cpu", weights_only=False)
        if ck.get("epoch", 0) == -1:
            print(f"[atla] egitim tamam: {ad}", flush=True)
            return
        print(f"[resume] {ad}", flush=True)
        YOLO(last).train(resume=True, workers=4)
        return
    print(f"[egitim] {ad}", flush=True)
    YOLO("yolov8n.pt").train(data=data, epochs=epochs, imgsz=640, batch=16,
                             patience=patience, workers=4, name=ad)


def main():
    once("1_merge", [PY, "merge_dataset_separate.py"])
    once("2_aug_cukur", [PY, "augment_dataset.py", "model_cukur", "1", "1"])
    once("3_aug_cop", [PY, "augment_dataset.py", "model_cop", "6", "2"])
    once("4_aug_tabela", [PY, "augment_dataset.py", "model_tabela", "4", "0.5"])
    egit("final_cukur", "dataset/model_cukur/data.yaml", 30, 10)
    egit("final_cop", "dataset/model_cop/data.yaml", 40, 12)
    egit("final_tabela", "dataset/model_tabela/data.yaml", 30, 10)
    print("[bitti] tum egitimler tamamlandi", flush=True)


if __name__ == "__main__":
    main()
