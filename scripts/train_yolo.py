# scripts/train_yolo.py
import argparse
import shutil
from pathlib import Path
from datetime import datetime

from ultralytics import YOLO


def export_named_model(run_dir: Path, model_name: str) -> Path | None:
    run_dir = Path(run_dir)
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        print(f"[WARN] best.pt not found at: {best}")
        return None

    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{model_name}.pt"
    shutil.copy(best, out)
    print(f"QS_MODEL_SAVED={out}")  # UI can parse this if needed
    return out


def train_yolo(data: Path, model: Path, epochs: int, imgsz: int, batch: int, name: str):
    data = Path(data)
    model = Path(model)

    if not data.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data}")
    if not model.exists():
        raise FileNotFoundError(f"Base model not found: {model}")

    print("Starting YOLO training...")
    print(f"Data:   {data}")
    print(f"Model:  {model}")
    print(f"Epochs: {epochs}, imgsz: {imgsz}, batch: {batch}")
    print(f"Name:   {name}")
    print("-" * 60)

    yolo = YOLO(str(model))

    # project/name -> produces runs/detect/<nameX>
    # We keep it stable: runs/detect/train  (as you already have)
    results = yolo.train(
        data=str(data),
        epochs=int(epochs),
        imgsz=int(imgsz),
        batch=int(batch),
        project="runs/detect",
        name="train",
        exist_ok=True,
        device="cpu",
        workers=0,
        verbose=True,
    )

    run_dir = Path(getattr(results, "save_dir", "")) if results is not None else None
    if not run_dir or not run_dir.exists():
        # fallback: default ultralytics path
        run_dir = Path("runs/detect/train")

    print(f"Results saved to {run_dir}")  # UI parses this line
    export_named_model(run_dir, name)


def build_argparser():
    p = argparse.ArgumentParser(description="Q-Sentinel TrainingModels - YOLO trainer")
    p.add_argument("--data", type=str, default=str(Path("configs") / "pu_dataset.yaml"),
                   help="Path to YOLO dataset yaml")
    p.add_argument("--model", type=str, default=str(Path("scripts") / "yolov8n.pt"),
                   help="Path to base model .pt (yolov8n.pt etc.)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--name", type=str, default=f"qs_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                   help="Export name for final model (models/<name>.pt)")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train_yolo(
        data=Path(args.data),
        model=Path(args.model),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name
    )
