"""3 ayri tek-sinifli uzman modeli sirayla egitir (bkz. merge_dataset_separate.py).
Ayni hiperparametreler (80 epoch, 640px, batch 16, yolov8n) - tek fark
her modelin SADECE kendi sinifina ait veriyi gormesi."""
from ultralytics import YOLO

EPOCHS = 80
IMG_SIZE = 640
BATCH_SIZE = 16
BASE_MODEL = "yolov8n.pt"
PATIENCE = 25

JOBS = [
    ("dataset/model_cukur/data.yaml", "uzman_cukur"),
    ("dataset/model_tabela/data.yaml", "uzman_tabela"),
    ("dataset/model_cop/data.yaml", "uzman_cop"),
]


def main():
    for data_yaml, name in JOBS:
        print(f"\n===== EGITIM BASLIYOR: {name} ({data_yaml}) =====\n")
        model = YOLO(BASE_MODEL)
        model.train(
            data=data_yaml,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            patience=PATIENCE,
            name=name,
        )
        print(f"\n===== EGITIM BITTI: {name} =====\n")


if __name__ == "__main__":
    main()
