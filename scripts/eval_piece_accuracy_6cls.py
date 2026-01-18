from pathlib import Path
import yaml
import cv2
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

NOK_IDS = {3, 4, 5}  # PU1_NOK, PU2_NOK, PU3_NOK

def parse_label_ids(label_path: Path):
    if not label_path.exists():
        return []
    lines = label_path.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
    ids = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        try:
            ids.append(int(float(parts[0])))
        except:
            pass
    return ids

def is_piece_nok_from_ids(ids):
    return any(i in NOK_IDS for i in ids)

def eval_piece_accuracy(
    best_model_path="runs/detect/train/weights/best.pt",
    dataset_yaml="configs/pu_dataset.yaml",
    conf=0.4,
    imgsz=640,
    device="cpu",
):
    ROOT = Path(__file__).resolve().parent.parent
    model_path = ROOT / best_model_path
    yaml_path = ROOT / dataset_yaml

    cfg = yaml.safe_load(yaml_path.read_text())
    base = Path(cfg["path"])
    val_images = base / cfg["val"]
    val_labels = base / "labels" / "val"

    model = YOLO(str(model_path))

    images = sorted([p for p in val_images.rglob("*") if p.suffix.lower() in IMG_EXTS])
    if not images:
        raise RuntimeError(f"No val images in {val_images}")

    TP = FP = TN = FN = 0  # NOK is "positive"

    for img_path in images:
        gt_ids = parse_label_ids(val_labels / f"{img_path.stem}.txt")
        gt_nok = is_piece_nok_from_ids(gt_ids)

        img = cv2.imread(str(img_path))
        pred = model.predict(img, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
        pred_ids = [int(x) for x in pred.boxes.cls.tolist()] if pred and len(pred.boxes) else []
        pred_nok = is_piece_nok_from_ids(pred_ids)

        if gt_nok and pred_nok: TP += 1
        elif (not gt_nok) and pred_nok: FP += 1
        elif (not gt_nok) and (not pred_nok): TN += 1
        elif gt_nok and (not pred_nok): FN += 1

    total = TP + FP + TN + FN
    acc = (TP + TN) / total if total else 0
    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0

    print("\nPiece-level OK/NOK evaluation on VAL (NOK=positive):")
    print(f"Total: {total}")
    print(f"TP (NOK correctly flagged): {TP}")
    print(f"FP (false reject OK->NOK): {FP}")
    print(f"TN (OK correctly passed):  {TN}")
    print(f"FN (missed NOK):           {FN}")
    print(f"\nAccuracy:  {acc*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")

if __name__ == "__main__":
    eval_piece_accuracy()
