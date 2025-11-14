#!/usr/bin/env python3
"""
run_adas_demo.py

Improved ADAS demo runner:
- loads YOLO detection and optional segmentation weights (Ultralytics)
- optional DeepSORT tracking
- lane detection (segmentation-assisted)
- traffic light color detector (optional)
- writes annotated video and events CSV
- uses YAML config (configs/paths.yaml) if present

Usage examples:
  python scripts/run_adas_demo.py --video "E:/projects/riding_bike.mp4" --output "outputs/adas_demo.mp4" --seg "run_yolov8n_seg2/weights/best.pt" --device auto --tl
  python scripts/run_adas_demo.py --config configs/paths.yaml --tl
"""

from pathlib import Path
import argparse
import yaml
import time
import csv
import math
import logging
import sys
import collections
import traceback

import cv2
import numpy as np

# optional imports
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except Exception:
    DeepSort = None

# ---------------------------
# Default parameters
# ---------------------------
DEFAULT_CONFIG = {
    "detect_weights": "yolov8n.pt",
    "seg_weights": None,
    "device": "auto",  # 'auto'|'cpu'|'cuda'
    "conf": 0.3,
    "output_dir": "outputs",
    "tl_detect": False,
    "track_enable": True,
    "seg_enable": False,
    "log_file": None
}

# ---------------------------
# Helpers: logging, config
# ---------------------------
def setup_logging(level=logging.INFO, log_file: Path = None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        fh = logging.FileHandler(str(log_file), mode='w', encoding='utf-8')
        handlers.append(fh)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers
    )

def load_yaml_config(path: Path):
    if not path.exists():
        logging.warning("Config file not found: %s", path)
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def choose_device(requested: str):
    if requested == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return requested

# ---------------------------
# Model loading
# ---------------------------
def load_models(detect_weights: str, seg_weights: str, device: str):
    det_model = None
    seg_model = None
    if YOLO is None:
        logging.warning("ultralytics YOLO not installed. Detection disabled.")
    else:
        try:
            logging.info("Loading detection model: %s", detect_weights)
            det_model = YOLO(detect_weights) if detect_weights else None
        except Exception as e:
            logging.exception("Failed loading detection model: %s", e)
            det_model = None

    if seg_weights:
        if YOLO is None:
            logging.warning("ultralytics not present - segmentation disabled.")
        else:
            try:
                logging.info("Loading segmentation model: %s", seg_weights)
                seg_model = YOLO(seg_weights)
            except Exception as e:
                logging.exception("Failed loading segmentation model: %s", e)
                seg_model = None

    return det_model, seg_model

# ---------------------------
# Small CV helpers
# ---------------------------
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
def mask_from_hsv(hsv, low, high):
    low = np.array(low, dtype=np.uint8); high = np.array(high, dtype=np.uint8)
    m = cv2.inRange(hsv, low, high)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, KERNEL)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, KERNEL)
    return m

def detect_tl_color_simple(frame, roi_bottom_frac=0.45, min_area=40):
    """Return (state, bbox, score) for red/yellow/green or (None,None,0)"""
    H,W = frame.shape[:2]
    y2 = int(H * roi_bottom_frac)
    roi = frame[0:y2, :, :].copy()
    if roi.size == 0:
        return None, None, 0.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    red1 = mask_from_hsv(hsv, (0, 90, 60), (10, 255, 255))
    red2 = mask_from_hsv(hsv, (160, 90, 60), (179, 255, 255))
    red = cv2.bitwise_or(red1, red2)
    yellow = mask_from_hsv(hsv, (15, 100, 100), (40, 255, 255))
    green = mask_from_hsv(hsv, (40, 60, 60), (100, 255, 255))

    def best(mask):
        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for c in cnts:
            area = cv2.contourArea(c)
            if area < min_area: continue
            x,y,w,h = cv2.boundingRect(c)
            score = area
            if best is None or score > best[0]:
                best = (score, (x,y,w,h))
        return best

    br = best(red); by = best(yellow); bg = best(green)
    if br:
        s,b = br; x,y,w,h = b; return "red", (x,y,x+w,y+h), float(s)
    if by:
        s,b = by; x,y,w,h = b; return "yellow", (x,y,x+w,y+h), float(s)
    if bg:
        s,b = bg; x,y,w,h = b; return "green", (x,y,x+w,y+h), float(s)
    return None, None, 0.0

# ---------------------------
# Lane detection (simple, segmentation-assisted)
# ---------------------------
def lane_mask_from_seg(seg_mask, road_classes=(0,)):
    if seg_mask is None:
        return None
    mask = np.zeros_like(seg_mask, dtype=np.uint8)
    for rc in road_classes:
        mask[seg_mask == rc] = 255
    return mask

def refined_lane_detection(frame, seg_mask=None, roi_top_frac=0.62):
    """Return left_line, right_line, lane_center, debug"""
    H,W = frame.shape[:2]
    top = int(H * roi_top_frac)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    mask_roi = np.zeros_like(edges)
    cv2.rectangle(mask_roi, (0, top), (W-1, H-1), 255, -1)
    edges = cv2.bitwise_and(edges, mask_roi)

    if seg_mask is not None:
        try:
            if seg_mask.shape != edges.shape:
                seg_mask_res = cv2.resize((seg_mask>0).astype(np.uint8)*255, (W, H), interpolation=cv2.INTER_NEAREST)
            else:
                seg_mask_res = (seg_mask>0).astype(np.uint8)*255
            edges = cv2.bitwise_and(edges, seg_mask_res)

        except Exception:
            pass

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=40, maxLineGap=20)
    lefts=[]; rights=[]
    if lines is not None:
        for l in lines:
            x1,y1,x2,y2 = l[0]
            if x2==x1: continue
            slope = (y2-y1)/(x2-x1)
            if abs(slope) < 0.3: continue
            if slope < 0:
                lefts.append((x1,y1,x2,y2))
            else:
                rights.append((x1,y1,x2,y2))
    def avg_line(lines):
        if not lines: return None
        xs1 = np.array([l[0] for l in lines]); ys1 = np.array([l[1] for l in lines])
        xs2 = np.array([l[2] for l in lines]); ys2 = np.array([l[3] for l in lines])
        return int(xs1.mean()), int(ys1.mean()), int(xs2.mean()), int(ys2.mean())

    left = avg_line(lefts); right = avg_line(rights)
    center = None
    if left and right:
        cx = int((left[0]+right[0]) / 2); cy = int((left[1]+right[1]) / 2)
        center = ((left[0], left[1]), (right[0], right[1]))
    return left, right, center, {"edges":edges}

# ---------------------------
# Frame processing function
# ---------------------------
def process_frame(frame, det_model, seg_model, tracker, cfg, state):
    """
    Processes a single frame.
    Returns annotated frame, events_list (may be empty), updated state
    state: dict for tracker stateful data
    """
    H,W = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    seg_mask = None
    rdet = None

    if seg_model is not None and cfg.get("seg_enable", False):
        try:
            rseg = seg_model.predict(frame_rgb, device=cfg["device"], imgsz=640, verbose=False)[0]
            # extract masks or class map
            if hasattr(rseg, "masks") and rseg.masks is not None:
                arr = np.array(rseg.masks.data if hasattr(rseg.masks, "data") else rseg.masks.numpy())
                if arr.ndim == 3:
                    seg_mask = np.argmax(arr, axis=0).astype(np.int32)
                else:
                    seg_mask = arr.astype(np.int32)
        except Exception:
            seg_mask = None

    detections_for_tracker = []
    if det_model is not None:
        try:
            rdet = det_model.predict(frame_rgb, device=cfg["device"], imgsz=640, conf=cfg.get("conf",0.3), verbose=False)[0]
            if hasattr(rdet, "boxes") and rdet.boxes is not None:
                xyxy = rdet.boxes.xyxy.cpu().numpy() if hasattr(rdet.boxes, "xyxy") else np.array(rdet.boxes.xyxy)
                confs = rdet.boxes.conf.cpu().numpy() if hasattr(rdet.boxes, "conf") else np.array(rdet.boxes.conf)
                clsids = rdet.boxes.cls.cpu().numpy() if hasattr(rdet.boxes, "cls") else np.array(rdet.boxes.cls)
                for (x1,y1,x2,y2), conf, clsid in zip(xyxy, confs, clsids):
                    detections_for_tracker.append(([float(x1), float(y1), float(x2), float(y2)], float(conf), int(clsid)))
        except Exception as e:
            logging.debug("Detection error: %s", e)

    # Tracking
    tracks = []
    if tracker is not None and detections_for_tracker:
        try:
            tracks = tracker.update_tracks(detections_for_tracker, frame=frame)
        except Exception as ex:
            logging.warning("Tracker update failed: %s", ex)
            tracks = []

    # Lane detection
    left_line, right_line, lane_center, debug = refined_lane_detection(frame, seg_mask=seg_mask)

    # Traffic light (color) detection - simple fallback
    tl_state, tl_box, tl_score = (None, None, 0.0)
    if cfg.get("tl_detect", False):
        st, box, score = detect_tl_color_simple(frame)
        tl_state, tl_box, tl_score = st, box, score

    # Draw visuals and collect events
    vis = frame.copy()
    # draw lane
    if left_line:
        cv2.line(vis, (left_line[0],left_line[1]), (left_line[2],left_line[3]), (0,255,0), 2)
    if right_line:
        cv2.line(vis, (right_line[0],right_line[1]), (right_line[2],right_line[3]), (0,255,0), 2)
    if lane_center:
        (c1,c2) = lane_center
        cv2.line(vis, (c1[0],c1[1]), (c2[0],c2[1]), (255,0,0), 2)

    events = []
    # draw tracker boxes and compute pixel speed/TTC fallback
    now = time.time()
    for tr in tracks:
        try:
            if hasattr(tr, "is_confirmed") and not tr.is_confirmed():
                continue
        except Exception:
            pass
        try:
            tid = int(tr.track_id)
        except Exception:
            tid = abs(hash(str(tr))) % 10000
        try:
            ltrb = tr.to_ltrb()
        except Exception:
            try:
                ltrb = tr.to_tlbr()
            except Exception:
                continue
        x1,y1,x2,y2 = map(int, ltrb)
        cx = (x1+x2)//2; cy = y2
        # pixel speed fallback
        last = state["last_centroid"].get(tid, None)
        speed_px_s = 0.0
        if last is not None:
            px,py,t_prev = last
            dt = max(1e-6, now - t_prev)
            speed_px_s = math.hypot(cx-px, cy-py)/dt
        state["last_centroid"][tid] = (cx,cy,now)
        # TTC estimation (very rough: distance in pixels / speed_px_s)
        ttc = (frame.shape[0] - y2) / (speed_px_s + 1e-6) if speed_px_s>1e-6 else float("inf")
        color = (0,0,255) if ttc < 3.0 else (int( (hash(str(tid))%200)+30 ), 180, 30)
        cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)
        label = f"ID:{tid} {int(speed_px_s)}px/s"
        if ttc != float("inf"):
            label += f" TTC:{ttc:.1f}s"
        cv2.putText(vis, label, (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

        events.append({
            "time": now,
            "track_id": tid,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "speed_px_s": speed_px_s, "ttc_s": ttc
        })

    # draw TL
    if tl_box is not None:
        x1,y1,x2,y2 = map(int, tl_box)
        col = (0,0,255) if tl_state=="red" else (0,255,255) if tl_state=="yellow" else (0,255,0)
        cv2.rectangle(vis, (x1,y1), (x2,y2), col, 2)
        cv2.putText(vis, f"TL:{tl_state}", (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

    # HUD
    cv2.putText(vis, f"TL:{tl_state if tl_state else 'none'} Frame:{state['frame_id']}", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    state['frame_id'] += 1
    return vis, events, state

# ---------------------------
# Main runner
# ---------------------------
def run(video_path, output_path, csv_path, cfg):
    # prepare folders
    outdir = Path(output_path).parent
    outdir.mkdir(parents=True, exist_ok=True)
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

    device = choose_device(cfg.get("device","auto"))
    cfg["device"] = device
    logging.info("Using device: %s", device)

    det_model, seg_model = load_models(cfg.get("detect_weights"), cfg.get("seg_weights"), device)
    if cfg.get("seg_enable") and seg_model is None:
        logging.warning("seg_enable set but segmentation model not loaded; continuing without seg.")

    tracker = None
    if DeepSort is None:
        logging.warning("deep_sort_realtime not installed; tracking disabled.")
    elif cfg.get("track_enable", True):
        try:
            tracker = DeepSort(max_age=30)
        except Exception as e:
            logging.warning("Failed to init DeepSort: %s", e)
            tracker = None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(output_path), fourcc, FPS, (W, H))

    state = {"last_centroid": {}, "frame_id": 0}
    events_all = []
    frame_count = 0
    t0 = time.time()
    logging.info("Starting processing video: %s", video_path)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            vis, events, state = process_frame(frame, det_model, seg_model, tracker, cfg, state)
            # write
            writer.write(vis)
            events_all.extend([{"frame": state["frame_id"], **e} for e in events])
            frame_count += 1
            # lightweight progress
            if frame_count % 50 == 0:
                elapsed = time.time() - t0
                fps_est = frame_count / (elapsed + 1e-6)
                logging.info("Processed %d frames. FPS ~ %.2f", frame_count, fps_est)

    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
    except Exception as e:
        logging.exception("Fatal error during processing: %s", e)
        traceback.print_exc()
    finally:
        cap.release()
        writer.release()
        # save CSV
        keys = ["frame","time","track_id","x1","y1","x2","y2","speed_px_s","ttc_s"]
        try:
            with open(str(csv_path), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for r in events_all:
                    row = {k: r.get(k, None) for k in keys}
                    w.writerow(row)
            logging.info("Saved events CSV: %s", csv_path)
        except Exception as e:
            logging.exception("Failed to write CSV: %s", e)

    total_time = time.time() - t0
    logging.info("Done. Frames: %d, Time: %.1fs, Avg FPS: %.2f", frame_count, total_time, frame_count / (total_time + 1e-6))

# ---------------------------
# CLI / entrypoint
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(prog="run_adas_demo")
    p.add_argument("--config", type=str, help="YAML config with keys for detect_weights, seg_weights, output_dir, etc.")
    p.add_argument("--video", type=str, help="Input video path", required=False)
    p.add_argument("--output", type=str, help="Output annotated video path", required=False)
    p.add_argument("--csv", type=str, help="Output CSV path", required=False)
    p.add_argument("--detect", type=str, help="detection weights", default=None)
    p.add_argument("--seg", type=str, help="segmentation weights", default=None)
    p.add_argument("--device", type=str, default="auto", help="device: auto|cpu|cuda")
    p.add_argument("--conf", type=float, default=None, help="detection confidence threshold")
    p.add_argument("--tl", action="store_true", help="enable TL color detection")
    p.add_argument("--no-track", action="store_true", help="disable DeepSORT tracking")
    p.add_argument("--log", type=str, default=None, help="optional log file")
    return p.parse_args()

def merge_cfg(args):
    cfg = DEFAULT_CONFIG.copy()
    if args.config:
        y = load_yaml_config(Path(args.config))
        if isinstance(y, dict):
            cfg.update(y)
    # override with CLI args
    if args.detect:
        cfg["detect_weights"] = args.detect
    if args.seg:
        cfg["seg_weights"] = args.seg
    if args.device:
        cfg["device"] = args.device
    if args.conf is not None:
        cfg["conf"] = args.conf
    if args.tl:
        cfg["tl_detect"] = True
    if args.no_track:
        cfg["track_enable"] = False
    if args.log:
        cfg["log_file"] = args.log

    # finalize output paths
    outdir = Path(cfg.get("output_dir","outputs"))
    outdir.mkdir(parents=True, exist_ok=True)
    if args.output:
        out_video = Path(args.output)
    else:
        out_video = outdir / "adas_output.mp4"
    if args.csv:
        out_csv = Path(args.csv)
    else:
        out_csv = outdir / "adas_events.csv"

    cfg["output_video"] = str(out_video)
    cfg["output_csv"] = str(out_csv)
    return cfg

if __name__ == "__main__":
    args = parse_args()
    cfg = merge_cfg(args)
    setup_logging(level=logging.INFO, log_file=Path(cfg["log_file"]) if cfg.get("log_file") else None)
    logging.info("Configuration: %s", {k:v for k,v in cfg.items() if k not in ('detect_weights','seg_weights')})
    run(args.video or cfg.get("video", ""), cfg["output_video"], cfg["output_csv"], cfg)
