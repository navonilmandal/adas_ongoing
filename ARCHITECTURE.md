# ADAS Project — Architecture

> Professional architecture and component design for the ADAS pipeline (YOLOv8 segmentation, detection, DeepSORT tracking, lane detection, TTC, traffic-light color detection, HUD & logger).

---

## 1. High-level overview

This ADAS system processes dashcam video (or live camera stream) and produces a safety HUD + event logs.  
It is modular: **Data → Training → Inference Pipeline → Postprocessing → Storage / UI**.

Key runtime components:
- **Frame Capture** (Video/Camera)
- **Preprocessing** (resize, color, calibration/homography)
- **Per-frame Inference**
  - Object **Detection** (YOLOv8)
  - **Segmentation** (YOLOv8-seg or same model)
  - **Traffic-Light Classifier** (color-based or model)
- **Tracking** (DeepSORT)
- **Classical CV modules**
  - Lane detection (edge + Hough / polynomial fit)
  - Hood removal (segmentation-based)
  - Speed & TTC computation (pixel-to-meter via calibration/homography)
- **Event Manager / Alarms** (lane departure, TTC alarm, TL events)
- **Output Writers**
  - Video with HUD (H.264)
  - CSV logs / JSON events
  - Optional: MQTT / REST endpoint for telemetry

Non-functional:
- Real-time (target ≥ 10–25 FPS on RTX3050)
- Deterministic logging for offline analysis
- GPU acceleration for inference; CPU fallback for edge deployment

---

## 2. Mermaid architecture diagram

> Paste this into a GitHub Markdown file — GitHub renders Mermaid natively.

```mermaid
flowchart LR
  A[Video / Camera Input] --> B[Frame Preprocessing]
  B --> C{Per-frame Inference}
  subgraph Inference
    C1[YOLOv8 Detection] --> C
    C2[YOLOv8 Segmentation] --> C
    C3[TL Color Detector] --> C
  end
  C --> D[Post-processing]
  D --> E[Tracking (DeepSORT)]
  D --> F[Lane Detection + Hood Removal]
  E --> G[Speed & TTC Calculator]
  F --> G
  G --> H[Event Manager & Alarms]
  H --> I[HUD Renderer]
  I --> J[Output Video / Frames]
  H --> K[CSV / JSON Logs]
  H --> L[Realtime API (REST/MQTT)]
  subgraph Training Pipeline
    M[Raw IDD dataset] --> N[Augmentation & Preproc]
    N --> O[YOLOv8 Training (seg/det)]
    O --> P[Model Weights (best.pt)]
  end
  O --> C1
  O --> C2
