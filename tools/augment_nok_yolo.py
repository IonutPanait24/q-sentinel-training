import os
import random
from pathlib import Path
import cv2
import numpy as np

# ---- CONFIG ----
DATASET = Path(r"C:\Users\yonut\PycharmProjects\Dataset")
IMG_DIR = DATASET / "images" / "train"
LBL_DIR = DATASET / "labels" / "train"

# classes NOK in your mapping
NOK_IDS = {3, 4, 5}  # PU1_NOK, PU2_NOK, PU3_NOK

# how many augmented copies per NOK image (adjust)
AUG_PER_IMAGE = 1  # start with 1; can increase to 2

# random seed for reproducibility
random.seed(42)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_yolo_labels(txt_path: Path):
    """Return list of (cls, xc, yc, w, h) normalized floats."""
    rows = []
    if not txt_path.exists():
        return rows
    for line in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        xc, yc, w, h = map(float, parts[1:5])
        rows.append((cls, xc, yc, w, h))
    return rows


def write_yolo_labels(txt_path: Path, rows):
    lines = []
    for cls, xc, yc, w, h in rows:
        # clamp to [0,1]
        xc = min(max(xc, 0.0), 1.0)
        yc = min(max(yc, 0.0), 1.0)
        w = min(max(w, 0.0), 1.0)
        h = min(max(h, 0.0), 1.0)
        lines.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def yolo_to_xyxy(row, W, H):
    cls, xc, yc, w, h = row
    x1 = (xc - w / 2) * W
    y1 = (yc - h / 2) * H
    x2 = (xc + w / 2) * W
    y2 = (yc + h / 2) * H
    return cls, x1, y1, x2, y2


def xyxy_to_yolo(cls, x1, y1, x2, y2, W, H):
    # clip
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W - 1, x2))
    y2 = max(0, min(H - 1, y2))
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    xc = x1 + w / 2
    yc = y1 + h / 2
    return (cls, xc / W, yc / H, w / W, h / H)


def apply_brightness_contrast(img):
    # brightness [-20..20], contrast [0.9..1.1]
    beta = random.uniform(-20, 20)
    alpha = random.uniform(0.9, 1.1)
    out = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return out


def apply_gamma(img):
    gamma = random.uniform(0.9, 1.2)
    inv = 1.0 / gamma
    table = (np.arange(256) / 255.0) ** inv * 255
    table = table.astype(np.uint8)
    return cv2.LUT(img, table)


def apply_blur(img):
    if random.random() < 0.5:
        k = random.choice([3, 5])
        return cv2.GaussianBlur(img, (k, k), 0)
    return img


def apply_noise(img):
    if random.random() < 0.5:
        noise = np.random.normal(0, random.uniform(3, 8), img.shape).astype(np.float32)
        out = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return out
    return img


def rotate_image_and_boxes(img, labels, angle_deg):
    H, W = img.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle_deg, 1.0)
    rotated = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    new_rows = []
    for row in labels:
        cls, x1, y1, x2, y2 = yolo_to_xyxy(row, W, H)

        # corners
        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        ones = np.ones((4, 1), dtype=np.float32)
        pts = np.hstack([corners, ones])
        pts2 = (M @ pts.T).T  # (4,2)

        nx1, ny1 = pts2[:, 0].min(), pts2[:, 1].min()
        nx2, ny2 = pts2[:, 0].max(), pts2[:, 1].max()

        # discard boxes that became tiny
        if (nx2 - nx1) < 2 or (ny2 - ny1) < 2:
            continue
        new_rows.append(xyxy_to_yolo(cls, nx1, ny1, nx2, ny2, W, H))

    return rotated, new_rows


def crop_and_boxes(img, labels, crop_pct):
    """Crop equally from all sides by crop_pct (0..0.05) and resize back to original."""
    H, W = img.shape[:2]
    dx = int(W * crop_pct)
    dy = int(H * crop_pct)
    x1c, y1c = dx, dy
    x2c, y2c = W - dx, H - dy

    cropped = img[y1c:y2c, x1c:x2c]
    if cropped.size == 0:
        return img, labels

    # adjust boxes for crop, then scale back
    new_rows = []
    for row in labels:
        cls, x1, y1, x2, y2 = yolo_to_xyxy(row, W, H)
        x1 -= x1c
        x2 -= x1c
        y1 -= y1c
        y2 -= y1c

        cw, ch = (x2c - x1c), (y2c - y1c)
        # clip to crop frame
        x1 = max(0, min(cw - 1, x1))
        x2 = max(0, min(cw - 1, x2))
        y1 = max(0, min(ch - 1, y1))
        y2 = max(0, min(ch - 1, y2))
        if (x2 - x1) < 2 or (y2 - y1) < 2:
            continue

        # resize back to original size
        sx = W / cw
        sy = H / ch
        x1 *= sx
        x2 *= sx
        y1 *= sy
        y2 *= sy

        new_rows.append(xyxy_to_yolo(cls, x1, y1, x2, y2, W, H))

    resized = cv2.resize(cropped, (W, H), interpolation=cv2.INTER_LINEAR)
    return resized, new_rows


def is_nok_image(label_rows):
    return any(cls in NOK_IDS for cls, *_ in label_rows)


def main():
    images = [p for p in IMG_DIR.iterdir() if p.suffix.lower() in IMG_EXTS]
    nok_images = []

    for img_path in images:
        lbl_path = LBL_DIR / f"{img_path.stem}.txt"
        rows = read_yolo_labels(lbl_path)
        if rows and is_nok_image(rows):
            nok_images.append((img_path, lbl_path, rows))

    if not nok_images:
        print("No NOK images found in train.")
        return

    print(f"Found NOK train images: {len(nok_images)}")
    created = 0

    for img_path, lbl_path, rows in nok_images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        for k in range(AUG_PER_IMAGE):
            aug = img.copy()
            new_rows = list(rows)

            # safe color/noise transforms
            aug = apply_brightness_contrast(aug)
            aug = apply_gamma(aug)
            aug = apply_blur(aug)
            aug = apply_noise(aug)

            # small rotation
            if random.random() < 0.7:
                angle = random.uniform(-2.0, 2.0)
                aug, new_rows = rotate_image_and_boxes(aug, new_rows, angle)

            # small crop
            if random.random() < 0.7:
                crop_pct = random.uniform(0.0, 0.04)
                aug, new_rows = crop_and_boxes(aug, new_rows, crop_pct)

            if not new_rows:
                continue

            out_stem = f"{img_path.stem}_aug{created}"
            out_img = IMG_DIR / f"{out_stem}.jpg"
            out_lbl = LBL_DIR / f"{out_stem}.txt"

            cv2.imwrite(str(out_img), aug)
            write_yolo_labels(out_lbl, new_rows)
            created += 1

    print(f"✅ Created augmented NOK samples: {created}")
    print("They were added directly to train/images and train/labels.")

if __name__ == "__main__":
    main()
