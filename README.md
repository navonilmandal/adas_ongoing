🚘 Advanced Driver Assistance System (ADAS) — Computer Vision Based

A modular, real-time ADAS suite built using YOLOv8 segmentation, DeepSORT tracking, lane detection, speed estimation, TTC warnings and AI-powered HUD.

📌 Overview

This project implements a real-time ADAS pipeline using computer vision, deep learning and classical image processing.
The system works on dashcam / bike / car videos and provides essential road-safety features such as:

Lane detection & departure warnings

Object detection with segmentation

Multi-object tracking

Speed & relative distance estimation

Time-To-Collision (TTC) warnings

On-screen HUD visualization

Traffic-light detection (color-based prototype)

The project was built and tested on Indian roads using the IDD-Lite dataset.

🚀 Features Implemented
✔ 1. Semantic Segmentation (YOLOv8-Seg)

Custom-trained YOLOv8n-seg model on IDD-Lite

7-class segmentation (drivable, non-drivable, vehicles, pedestrians, etc.)

✔ 2. Object Tracking (DeepSORT)

Unique ID assignment

Smooth tracking across frames

Re-identification embeddings

✔ 3. Speed Estimation / Relative Velocity

Pixel-to-meter calibration

Frame-to-frame object displacement

Speed output in km/h

CSV export

✔ 4. Time-To-Collision (TTC)

TTC = distance / relative speed

Real-time warning system

Alerts displayed in HUD

Event logging

✔ 5. Lane Detection

Canny edge + Hough lines

Segmentation-based hood removal

Lane departure detection

Lane deviation warning

✔ 6. ADAS HUD Overlay

Includes overlay elements:

TTC in seconds

Object ID with speed

Bounding boxes with segmentation colors

Lane centerline

Warning text

✔ 7. Traffic Light Detection (Prototype)

Color-based (Red / Yellow / Green)

Upper-ROI filtering

Shape + area filtering

Non-blocking (optional)

🗂 Project Structure
adas_project/
│
├── notebooks/
│   ├── adas_lane.ipynb
│   ├── adas_tracking.ipynb
│   ├── adas_full_pipeline.ipynb
│
├── scripts/
│   ├── run_adas_demo.py
│   ├── lane_detector.py
│   ├── ttc_tracker.py
│   ├── speed_estimator.py
│   ├── tl_detector.py
│   └── segmentation_utils.py
│
├── configs/
│   ├── paths.yaml
│   └── model_config.yaml
│
├── run_yolov8n_seg2/
│   ├── weights/
│   ├── preds_viz/
│   ├── tracks_kmh_final.csv
│   ├── output_riding_bike.mp4
│   └── output_riding_bike_track_final.mp4
│
├── data/  (ignored)
│
├── .gitignore
├── .gitattributes
├── requirements.txt
└── README.md

⚙️ Installation
1️⃣ Create Conda environment
conda create -n adas python=3.10 -y
conda activate adas

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Install torch (if missing)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

📥 Dataset Setup (IDD-Lite)

Download IDD-Lite:
https://idd.insaan.iiit.ac.in/dataset/details/

Place it here:

E:/projects/adas_local/data/idd_lite


Dataset is ignored in GitHub via .gitignore.

▶️ Running the ADAS Pipeline
1️⃣ Run segmentation + tracking + TTC + lane detection + HUD:
python scripts/run_adas_demo.py --video input.mp4

2️⃣ Output files:
output_riding_bike_track_kmh_final.mp4
tracks_kmh_final.csv
output_riding_bike_ttc.mp4

🛣 Roadmap (Planned Features)

 *Blind Spot Detection (BSD)

 *Automatic Helmet Detection

 *Curved lane polynomial fitting

 *Full Traffic Light Recognition (YOLO-based)

 *Ego-motion stabilization

 *MiDaS depth-based distance estimation

 *Driver monitoring system

 *Mobile app integration


 This project is released under the MIT License.
Feel free to use, modify, and distribute.

⭐ Acknowledgements

IDD Dataset

Ultralytics YOLOv8

DeepSORT Realtime

OpenCV

PyTorch

Navonil Mandal,Me :) 🚀
