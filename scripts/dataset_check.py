from pathlib import Path
from collections import Counter

DATASET = Path(r"C:\Users\yonut\PycharmProjects\Dataset")

def main():
    img_train = DATASET / "images" / "train"
    img_val = DATASET / "images" / "val"
    lab_train = DATASET / "labels" / "train"
    lab_val = DATASET / "labels" / "val"

    for p in [img_train, img_val, lab_train, lab_val]:
        if not p.exists():
            raise FileNotFoundError(f"Missing folder: {p}")

    def scan(img_dir, lab_dir):
        exts = {".jpg",".jpeg",".png",".bmp",".webp"}
        imgs = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in exts])

        missing = []
        class_counts = Counter()
        bad_lines = 0

        for im in imgs:
            txt = lab_dir / f"{im.stem}.txt"
            if not txt.exists():
                missing.append(im.name)
                continue

            lines = txt.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
            for ln in lines:
                ln = ln.strip()
                if not ln:
                    continue
                parts = ln.split()
                try:
                    cid = int(float(parts[0]))
                    if cid < 0 or cid > 5:
                        bad_lines += 1
                    class_counts[cid] += 1
                except:
                    bad_lines += 1

        return len(imgs), missing, class_counts, bad_lines

    t_imgs, t_missing, t_counts, t_bad = scan(img_train, lab_train)
    v_imgs, v_missing, v_counts, v_bad = scan(img_val, lab_val)

    print("\n=== TRAIN ===")
    print("Images:", t_imgs)
    print("Missing labels:", len(t_missing))
    if t_missing[:10]:
        print("Examples:", t_missing[:10])
    print("Class counts:", dict(t_counts))
    print("Bad lines:", t_bad)

    print("\n=== VAL ===")
    print("Images:", v_imgs)
    print("Missing labels:", len(v_missing))
    if v_missing[:10]:
        print("Examples:", v_missing[:10])
    print("Class counts:", dict(v_counts))
    print("Bad lines:", v_bad)

if __name__ == "__main__":
    main()
