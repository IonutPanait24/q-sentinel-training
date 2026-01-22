Q-Sentinel TrainingModels
GUI-Based YOLO Training & Evaluation Platform

Q-Sentinel TrainingModels is a desktop AI application that provides a complete, visual, and controlled workflow for training YOLO-based computer vision models.

It removes the need for:

manual CLI training

direct YAML editing

interpreting noisy console outputs

and replaces them with:

a modern GUI

guided training steps

real-time progress monitoring

automatic metrics visualization

This project is part of the Q-Sentinel AI Vision System and was developed as a final AI Engineering project.

🚀 Key Features

✔️ Dataset folder validation (images + labels)
✔️ Automatic YOLO data.yaml generation
✔️ Class detection & visualization
✔️ Configurable training parameters (epochs, batch, image size)
✔️ Live training logs (YOLO output)
✔️ Epoch-based progress tracking
✔️ KPI metrics (Precision, Recall, mAP)
✔️ Loss & metrics charts
✔️ Training artifact preview
✔️ Safe start / stop training control

🧠 Why This Project Matters

Most YOLO training workflows are:

CLI-only

error-prone

hard to debug

unfriendly for non-ML users

Q-Sentinel TrainingModels bridges the gap between:

raw ML tooling

real-world engineering usability

It demonstrates how AI systems can be production-ready, user-safe, and visual.

🖥️ Application Overview
1️⃣ Dataset Tab

Dataset folder selection

Dataset structure validation

Automatic data.yaml generation

Detected classes preview

Dataset KPIs:

TRAIN images

VAL images

Missing labels

Invalid label lines

2️⃣ Training Tab

Base YOLO model selection (.pt)

Training configuration:

epochs

image size

batch size

Controlled training start / stop

Live YOLO logs

Training progress tracking

3️⃣ Charts Tab

Training KPIs:

Precision

Recall

mAP50

mAP50-95

Charts:

train / validation loss curves

metrics evolution per epoch

Training artifacts:

results curves

confusion matrix

🧱 System Architecture
Dataset (images + labels)
        ↓
Dataset Validation
        ↓
data.yaml Generation
        ↓
YOLO Training (Ultralytics)
        ↓
runs/detect/trainX
        ↓
Metrics • Charts • Artifacts

🛠️ Tech Stack
Core

Python 3.11+

PySide6 (Qt for Python)

AI / ML

YOLOv8 (Ultralytics)

PyTorch

Data & Visualization

Pandas

Matplotlib

Engineering

Subprocess-controlled training

Real-time log parsing

Thread-safe UI updates

Modular architecture (UI / Worker / Metrics)

📁 Project Structure
q-sentinel-training/
│
├── ui/
│   ├── app.py                 # main GUI application
│   ├── training_worker.py     # subprocess & progress parser
│   ├── metrics.py             # results.csv processing
│   ├── curves.py              # charts generation
│   ├── widgets.py             # reusable UI components
│   └── styles.py              # dark tech UI theme
│
├── scripts/
│   ├── train_yolo.py          # YOLO training wrapper
│   ├── dataset_check.py       # dataset validator
│   └── generate_yaml.py       # auto data.yaml generator
│
├── configs/
├── runs/                      # YOLO outputs (auto-generated)
├── models/                    # exported models
├── assets/
├── requirements.txt
└── README.md

▶️ Getting Started
1. Install dependencies
pip install -r requirements.txt

2. Run the application
python ui/app.py

📤 Model Output

After training:

runs/detect/trainX/weights/best.pt


This model can be:

used directly for inference

exported to production

integrated into Q-Sentinel Runtime

🎓 Educational Context

Developed as part of:

AI Engineering – Software Development Academy (SDA)

Demonstrates:

end-to-end ML workflows

applied computer vision

production-oriented AI design

GUI-driven ML systems

🧩 Part of the Q-Sentinel Ecosystem

Q-Sentinel TrainingModels → training & evaluation

Q-Sentinel Runtime → inference & monitoring

Together they form a complete AI Vision System.

🔮 Future Improvements

Model export manager

Multi-GPU support

Experiment comparison

Inference preview

Dataset augmentation tools

👤 Author

[Panait Ionut]
AI Engineering Graduate
Software Development Academy (SDA)