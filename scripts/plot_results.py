from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def plot_results():
    ROOT = Path(__file__).resolve().parent.parent
    csv_path = ROOT / "runs" / "detect" / "train" / "results.csv"
    out_dir = ROOT / "runs" / "detect" / "train"

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}. Run training first.")

    df = pd.read_csv(csv_path)

    def find_col(key):
        key = key.lower()
        for c in df.columns:
            if key in c.lower():
                return c
        return None

    box_loss = find_col("train/box_loss")
    cls_loss = find_col("train/cls_loss")
    dfl_loss = find_col("train/dfl_loss")

    map50 = find_col("metrics/mAP50")
    map5095 = find_col("metrics/mAP50-95") or find_col("metrics/mAP50_95")
    prec = find_col("metrics/precision")
    rec = find_col("metrics/recall")

    # Loss
    plt.figure()
    if box_loss: plt.plot(df[box_loss], label="box_loss")
    if cls_loss: plt.plot(df[cls_loss], label="cls_loss")
    if dfl_loss: plt.plot(df[dfl_loss], label="dfl_loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    p1 = out_dir / "custom_loss.png"
    plt.savefig(p1)

    # Metrics
    plt.figure()
    if map50: plt.plot(df[map50], label="mAP50")
    if map5095: plt.plot(df[map5095], label="mAP50-95")
    if prec: plt.plot(df[prec], label="precision")
    if rec: plt.plot(df[rec], label="recall")
    plt.title("Validation Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    p2 = out_dir / "custom_metrics.png"
    plt.savefig(p2)

    print("Saved:")
    print(p1)
    print(p2)

if __name__ == "__main__":
    plot_results()
