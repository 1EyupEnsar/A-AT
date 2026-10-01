"""Cukur icin GERCEK PIKSEL SEGMENTASYONU modelini egitir (yolov8n-seg).
main.py'deki kutu-alani tabanli kaba siddet tahmininin yerini, bu modelle
uretilen GERCEK maske piksel alani alabilecek (bkz. merge_dataset_seg.py)."""
from ultralytics import YOLO


def main():
    model = YOLO("yolov8n-seg.pt")
    model.train(
        data="dataset/model_cukur_seg/data.yaml",
        epochs=80,
        imgsz=640,
        batch=16,
        patience=25,
        name="cukur_segmentasyon",
    )
    print("Segmentasyon egitimi tamamlandi.")


if __name__ == "__main__":
    main()
