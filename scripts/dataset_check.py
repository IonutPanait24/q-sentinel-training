from __future__ import annotations

import argparse
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _collect_images(images_dir: Path):
    if not images_dir.exists():
        return []
    return [p for p in images_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]


def _check_split(images_dir: Path, labels_dir: Path, split_name: str):
    images = _collect_images(images_dir)

    missing_labels = 0
    bad_lines = 0
    class_counts = {}

    for img in images:
        txt = labels_dir / f"{img.stem}.txt"
        if not txt.exists():
            missing_labels += 1
            continue

        try:
            lines = txt.read_text(encoding="utf-8", errors="replace").splitlines()
            for ln in lines:
                ln = ln.strip()
                if not ln:
                    continue
                parts = ln.split()
                # YOLO format: cls x y w h
                if len(parts) != 5:
                    bad_lines += 1
                    continue
                cls_id = int(float(parts[0]))
                class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
        except Exception:
            bad_lines += 1

    print(f"=== {split_name.upper()} ===")
    print(f"Images: {len(images)}")
    print(f"Missing labels: {missing_labels}")
    print(f"Bad lines: {bad_lines}")
    print(f"Class counts: {class_counts}")
    print()

    return {
        "images": len(images),
        "missing_labels": missing_labels,
        "bad_lines": bad_lines,
        "class_counts": class_counts,
    }


def main():
    ap = argparse.ArgumentParser(description="Q-Sentinel TrainingModels - dataset checker (folder mode)")
    ap.add_argument("--root", type=str, required=True, help="Dataset root folder containing images/ and labels/")
    args = ap.parse_args()

    root = Path(args.root)

    train_img = root / "images" / "train"
    val_img = root / "images" / "val"
    train_lbl = root / "labels" / "train"
    val_lbl = root / "labels" / "val"

    # Basic structure validation
    errors = []
    if not train_img.exists(): errors.append(f"Missing folder: {train_img}")
    if not val_img.exists(): errors.append(f"Missing folder: {val_img}")
    if not train_lbl.exists(): errors.append(f"Missing folder: {train_lbl}")
    if not val_lbl.exists(): errors.append(f"Missing folder: {val_lbl}")

    if errors:
        for e in errors:
            print("[ERROR]", e)
        raise SystemExit(2)

    r1 = _check_split(train_img, train_lbl, "train")
    r2 = _check_split(val_img, val_lbl, "val")

    total_issues = r1["missing_labels"] + r2["missing_labels"] + r1["bad_lines"] + r2["bad_lines"]
    if total_issues > 0:
        raise SystemExit(2)

    print("Dataset OK ✅")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
