from ultralytics import YOLO
from pathlib import Path

def train_yolo(
    model_name="yolov8n.pt",
    epochs=2,
    imgsz=640,
    batch=8,
):
    ROOT = Path(__file__).resolve().parent.parent
    data_path = ROOT / "configs" / "pu_dataset.yaml"

    print("Starting YOLO training...")
    print(f"Model: {model_name}")
    print(f"Dataset: {data_path}")
    print(f"epochs={epochs} imgsz={imgsz} batch={batch}\n")

    model = YOLO(model_name)

    results = model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(ROOT / "runs" / "detect"),
        name="train",
        exist_ok=True,
        workers=0,   # Windows safety
    )

    print("\n✅ Training finished!\n")
    print("Final Detection Metrics (val):")
    print(f"mAP50:     {results.box.map50:.4f}")
    print(f"mAP50-95:  {results.box.map:.4f}")
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall:    {results.box.mr:.4f}")

    print("\nArtifacts:")
    print(ROOT / "runs" / "detect" / "train" / "weights" / "best.pt")
    print(ROOT / "runs" / "detect" / "train" / "results.csv")
    print(ROOT / "runs" / "detect" / "train" / "results.png")

if __name__ == "__main__":
    train_yolo()
