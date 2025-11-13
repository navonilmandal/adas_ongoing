#!/usr/bin/env python3
"""
run_adas_demo.py

Single-file ADAS demo runner:
- Loads detection (YOLO) and optional segmentation model
- Performs refined lane detection (segmentation-based hood removal)
- Runs DeepSORT tracking
- Estimates speed (pixel fallback) and TTC
- Detects traffic light color (optional)
- Writes annotated video and CSV events

Usage:
  python scripts/run_adas_demo.py --video /path/to/video.mp4 --output /path/to/out.mp4
"""

import argparse
from pathlib import Path
import time
import csv
import math
import collections
import sys

import cv2
import numpy as np

# Optional imports that must be installed in your environment
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except Exception:
    DeepSort = None

# -----------------------------
# DEFAULT CONFIG (change as needed)
# -----------------------------
DEFAULT_DETECT_WEIGHTS = "yolov8n.pt"
DEFAULT_SEG_WEIGHTS = None  # e.g. "run_yolov8n_seg2/weights/best.pt" or None to disable seg
DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_CONF = 0.3
DEVICE_DEFAULT = "cpu"  # change to "cuda" if GPU available and torch supports it

# Lane detection defaults
ROI_TOP_FRAC = 0.62
CENTER_SEARCH_W_FRAC = 0.5
CENTER_SEARCH_H_FRAC = 0.45
DARK_THRESH = 90
MIN_DARK_AREA = 2000
CANNY_LOW = 50
CANNY_HIGH = 150
HOUGH_THRESH = 30
MIN_LINE_LENGTH = 40
MAX_LINE_GAP = 20
SLOPE_MIN = 0.35

# Traffic-light color detector ROI (upper part)
TL_ROI_BOTTOM_FRAC = 0.45
MIN_TL_AREA = 40

# ADAS thresholds
TTC_THRESHOLD = 3.0
DEPARTURE_THRESHOLD_PX = 60
ALARM_COOLDOWN = 2.0
BEEP_ON_ALARM = False  # windows beep may fail on linux

# -----------------------------
# Utility functions
# -----------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_models(det_weights, seg_weights, device):
    det_model = None
    seg_model = None
    if YOLO is None:
        raise RuntimeError("ultralytics (YOLO) not installed in this environment.")
    det_model = YOLO(det_weights) if det_weights is not None else None
    if seg_weights:
        try:
            seg_model = YOLO(str(seg_weights))
        except Exception as e:
            print("Warning: failed to load segmentation model:", e, file=sys.stderr)
            seg_model = None
    return det_model, seg_model

# -----------------------------
# Lane detection (refined with segmentation-based hood removal)
# Returns left_line, right_line, lane_center, debug
# -----------------------------
def lane_mask_from_seg(seg_mask, road_classes={0}):
    """Convert segmentation label map into binary road mask (255 = road)."""
    if seg_mask is None:
        return None
    mask = np.zeros_like(seg_mask, dtype=np.uint8)
    for rc in road_classes:
        mask[seg_mask == rc] = 255
    return mask

def refined_lane_detection(frame, seg_mask=None,
                           roi_top_frac=ROI_TOP_FRAC,
                           center_search_w_frac=CENTER_SEARCH_W_FRAC,
                           center_search_h_frac=CENTER_SEARCH_H_FRAC,
                           dark_thresh=DARK_THRESH, min_dark_area=MIN_DARK_AREA,
                           canny_low=CANNY_LOW, canny_high=CANNY_HIGH,
                           hough_rho=1, hough_theta=np.pi/180, hough_thresh=HOUGH_THRESH,
                           min_line_length=MIN_LINE_LENGTH, max_line_gap=MAX_LINE_GAP,
                           slope_thresh_min=SLOPE_MIN):
    H, W = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    top_y = int(H * roi_top_frac)
    mask_roi = np.zeros_like(gray)
    poly = np.array([[(0,H),(0,top_y),(W,top_y),(W,H)]], dtype=np.int32)
    cv2.fillPoly(mask_roi, poly, 255)

    # fallback dark-region hood detection (in case seg_mask is None)
    sw = int(W * center_search_w_frac)
    sh = int(H * center_search_h_frac)
    sx1 = max(0, (W - sw)//2); sx2 = min(W, sx1 + sw)
    sy1 = max(top_y, H - sh); sy2 = H
    search_patch = gray[sy1:sy2, sx1:sx2] if sy2>sy1 else np.zeros((1,1), dtype=np.uint8)
    hood_box = None
    try:
        _, dark_mask = cv2.threshold(search_patch, dark_thresh, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)
        cnts, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cnts = sorted(cnts, key=lambda c: cv2.contourArea(c), reverse=True)
            for c in cnts:
                area = cv2.contourArea(c)
                if area < min_dark_area:
                    continue
                x,y,w,h = cv2.boundingRect(c)
                X1, Y1 = sx1 + x, sy1 + y
                X2, Y2 = X1 + w, Y1 + h
                pad_x = int(0.06 * W); pad_y = int(0.06 * H)
                X1 = max(0, X1 - pad_x); Y1 = max(0, Y1 - pad_y)
                X2 = min(W, X2 + pad_x); Y2 = min(H, Y2 + pad_y)
                hood_box = (X1, Y1, X2, Y2)
                break
    except Exception:
        hood_box = None

    edges = cv2.Canny(blur, canny_low, canny_high)
    edges_roi = cv2.bitwise_and(edges, mask_roi)

    # segmentation-based road mask to keep only road edges
    road_mask = lane_mask_from_seg(seg_mask)
    if road_mask is not None:
        if road_mask.shape != edges_roi.shape:
            road_mask = cv2.resize(road_mask, (edges_roi.shape[1], edges_roi.shape[0]), interpolation=cv2.INTER_NEAREST)
        edges_roi = cv2.bitwise_and(edges_roi, road_mask)

    # exclude hood area
    if hood_box is not None:
        X1,Y1,X2,Y2 = hood_box
        try:
            edges_roi[Y1:Y2, X1:X2] = 0
        except Exception:
            pass

    ker2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    edges_roi = cv2.morphologyEx(edges_roi, cv2.MORPH_CLOSE, ker2)

    raw = cv2.HoughLinesP(edges_roi, hough_rho, hough_theta, hough_thresh,
                          minLineLength=min_line_length, maxLineGap=max_line_gap)
    candidates = []
    if raw is not None:
        for l in raw:
            x1,y1,x2,y2 = l[0]
            if x2 == x1:
                slope = float('inf')
            else:
                slope = (y2-y1)/(x2-x1)
            if abs(slope) < slope_thresh_min:
                continue
            if hood_box is not None:
                hx1,hy1,hx2,hy2 = hood_box
                midx = (x1+x2)//2; midy = (y1+y2)//2
                if hx1 <= midx <= hx2 and hy1 <= midy <= hy2:
                    continue
            candidates.append((x1,y1,x2,y2,slope))

    left_lines = [ (x1,y1,x2,y2) for (x1,y1,x2,y2,s) in candidates if s < 0 ]
    right_lines = [ (x1,y1,x2,y2) for (x1,y1,x2,y2,s) in candidates if s > 0 ]

    def aggregate(group):
        if not group:
            return None
        xs1 = np.array([g[0] for g in group]); ys1 = np.array([g[1] for g in group])
        xs2 = np.array([g[2] for g in group]); ys2 = np.array([g[3] for g in group])
        return (int(np.median(xs1)), int(np.median(ys1)), int(np.median(xs2)), int(np.median(ys2)))

    left = aggregate(left_lines)
    right = aggregate(right_lines)
    lane_center = None
    if left is None and right is None:
        lane_center = None
    elif left is None:
        lx1,ly1,lx2,ly2 = right
        lane_center = ((lx1 - W//4, ly1), (lx2 - W//4, ly2))
    elif right is None:
        rx1,ry1,rx2,ry2 = left
        lane_center = ((rx1 + W//4, ry1), (rx2 + W//4, ry2))
    else:
        lx1,ly1,lx2,ly2 = left; rx1,ry1,rx2,ry2 = right
        lane_center = ((int((lx1+rx1)/2), int((ly1+ry1)/2)), (int((lx2+rx2)/2), int((ly2+ry2)/2)))

    debug = {"edges_roi": edges_roi, "hood_box": hood_box, "candidates": candidates}
    return left, right, lane_center, debug

# -----------------------------
# Traffic-light color detector (simple HSV)
# -----------------------------
def detect_tl_color(frame, roi_bottom_frac=TL_ROI_BOTTOM_FRAC, min_area=MIN_TL_AREA):
    H,W = frame.shape[:2]
    y1 = 0; y2 = int(H * roi_bottom_frac)
    roi = frame[y1:y2,:].copy()
    if roi.size == 0:
        return None, None, 0.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # ranges
    red1 = ((0, 90, 60), (10, 255, 255))
    red2 = ((160, 90, 60), (179, 255, 255))
    yellow = ((15, 100, 100), (40, 255, 255))
    green = ((40, 60, 60), (100, 255, 255))

    def mask_range(rng):
        low, high = rng
        low = np.array(low, dtype=np.uint8); high = np.array(high, dtype=np.uint8)
        m = cv2.inRange(hsv, low, high)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
        return m

    m_red = cv2.bitwise_or(mask_range(red1), mask_range(red2))
    m_y = mask_range(yellow)
    m_g = mask_range(green)

    def best_blob(mask):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for c in cnts:
            area = cv2.contourArea(c)
            if area < min_area: continue
            x,y,w,h = cv2.boundingRect(c)
            # simple aspect ratio filter
            aspect = float(w)/(h+1e-6)
            if aspect>2.5: continue
            score = area
            if best is None or score>best[0]:
                best = (score,(x,y,w,h))
        return best

    br = best_blob(m_red)
    by = best_blob(m_y)
    bg = best_blob(m_g)

    # priority: red > yellow > green
    if br is not None:
        _, bbox = br; x,y,w,h = bbox; return "red",(x,y,x+w,y+h), float(br[0])
    if by is not None:
        _, bbox = by; x,y,w,h = bbox; return "yellow",(x,y,x+w,y+h), float(by[0])
    if bg is not None:
        _, bbox = bg; x,y,w,h = bbox; return "green",(x,y,x+w,y+h), float(bg[0])
    return None, None, 0.0

# -----------------------------
# Main runner
# -----------------------------
def run(video_path, output_path, out_csv_path, detect_weights, seg_weights, device, conf_threshold, use_tl):
    video_path = Path(video_path)
    output_path = Path(output_path)
    out_csv_path = Path(out_csv_path)
    ensure_dir(output_path.parent)
    ensure_dir(out_csv_path.parent)

    print("Loading models...")
    det_model, seg_model = load_models(detect_weights, seg_weights, device)

    # Tracker
    if DeepSort is None:
        print("Warning: deep_sort_realtime not installed; tracking disabled.")
        tracker = None
    else:
        tracker = DeepSort(max_age=30)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {video_path}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(output_path), fourcc, FPS, (W, H))

    events = []
    last_centroid = {}
    last_alarm_time = collections.defaultdict(lambda: -9999)
    frame_id = 0
    t_start = time.time()
    print("Processing frames...")

    # Pre-check for model TL class presence (optional)
    model_has_tl_class = False
    if det_model is not None:
        try:
            # probe one frame to check names mapping
            retp, framep = cap.read()
            if retp:
                probe = det_model.predict(cv2.cvtColor(framep, cv2.COLOR_BGR2RGB), device=device, imgsz=640, conf=0.4, verbose=False)[0]
                if hasattr(probe, "names"):
                    names_map = probe.names
                    model_has_tl_class = any("traffic" in str(n).lower() for n in names_map.values())
                # rewind
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        except Exception:
            model_has_tl_class = False

    # palette
    palette = (np.random.RandomState(2).randint(0,255,(256,3))).astype(np.int32)
    PALETTE = [ (int(c[0]), int(c[1]), int(c[2])) for c in palette ]

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t_frame = time.time()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # segmentation mask (label map) extraction if seg_model provided
        seg_mask = None
        if seg_model is not None:
            try:
                rseg = seg_model.predict(frame_rgb, device=device, imgsz=640, verbose=False)[0]
                if hasattr(rseg, "masks") and rseg.masks is not None:
                    arr = np.array(rseg.masks.data if hasattr(rseg.masks, "data") else rseg.masks.numpy())
                    if arr.ndim == 3:
                        seg_mask = np.argmax(arr, axis=0).astype(np.int32)
                    else:
                        seg_mask = arr.astype(np.int32)
            except Exception:
                seg_mask = None

        # lane detection
        left_line, right_line, lane_center, debug_lane = refined_lane_detection(frame, seg_mask=seg_mask)

        # detection (YOLO)
        detections_for_tracker = []
        rdet = None
        if det_model is not None:
            try:
                rdet = det_model.predict(frame_rgb, device=device, imgsz=640, conf=conf_threshold, verbose=False)[0]
                if hasattr(rdet, "boxes") and rdet.boxes is not None:
                    xyxy = rdet.boxes.xyxy.cpu().numpy() if hasattr(rdet.boxes, "xyxy") else np.array(rdet.boxes.xyxy)
                    confs = rdet.boxes.conf.cpu().numpy() if hasattr(rdet.boxes, "conf") else np.array(rdet.boxes.conf)
                    clsids = rdet.boxes.cls.cpu().numpy() if hasattr(rdet.boxes, "cls") else np.array(rdet.boxes.cls)
                    for (x1,y1,x2,y2), conf, clsid in zip(xyxy, confs, clsids):
                        detections_for_tracker.append(([float(x1), float(y1), float(x2), float(y2)], float(conf), str(int(clsid))))
            except Exception as e:
                print("Detection error:", e, file=sys.stderr)

        # tracking
        tracks = []
        if tracker is not None and len(detections_for_tracker)>0:
            try:
                tracks = tracker.update_tracks(detections_for_tracker, frame=frame)
            except Exception:
                # fallback: no tracks
                tracks = []

        # traffic light detection (prefer trained model box if present)
        tl_state = None; tl_box = None
        if use_tl:
            # try model-based TL if model appears to have class
            if model_has_tl_class and rdet is not None:
                try:
                    names = rdet.names if hasattr(rdet, "names") else {}
                    xyxy = rdet.boxes.xyxy.cpu().numpy() if hasattr(rdet.boxes, "xyxy") else np.array(rdet.boxes.xyxy)
                    confs = rdet.boxes.conf.cpu().numpy() if hasattr(rdet.boxes, "conf") else np.array(rdet.boxes.conf)
                    clsids = rdet.boxes.cls.cpu().numpy() if hasattr(rdet.boxes, "cls") else np.array(rdet.boxes.cls)
                    best_conf = 0.0
                    for (x1,y1,x2,y2), conf, clsid in zip(xyxy, confs, clsids):
                        clsid = int(clsid)
                        name = str(names.get(clsid, "")).lower() if names is not None else ""
                        if "traffic" in name:
                            if conf > best_conf:
                                best_conf = float(conf)
                                tl_box = (int(x1), int(y1), int(x2), int(y2))
                    if tl_box is not None:
                        # color classification inside box
                        x1,y1,x2,y2 = tl_box
                        crop = frame[y1:y2, x1:x2]
                        if crop.size>0:
                            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                            h_mean = hsv[...,0].mean()
                            s_mean = hsv[...,1].mean()
                            v_mean = hsv[...,2].mean()
                            if (h_mean<15 or h_mean>160) and s_mean>80 and v_mean>50:
                                tl_state = "red"
                            elif 15<=h_mean<=40 and s_mean>90:
                                tl_state = "yellow"
                            elif 40<h_mean<=100 and s_mean>60:
                                tl_state = "green"
                            else:
                                tl_state = "unknown"
                except Exception:
                    tl_state = None; tl_box = None
            # fallback to color-based TL
            if tl_state is None:
                st, box, score = detect_tl_color(frame)
                if st is not None:
                    tl_state = st; tl_box = box

        # draw overlays
        vis = frame.copy()
        if left_line is not None:
            cv2.line(vis, (left_line[0],left_line[1]), (left_line[2],left_line[3]), (0,255,0), 3)
        if right_line is not None:
            cv2.line(vis, (right_line[0],right_line[1]), (right_line[2],right_line[3]), (0,255,0), 3)
        if lane_center is not None:
            (c1,c2) = lane_center
            cv2.line(vis, (c1[0],c1[1]), (c2[0],c2[1]), (255,0,0), 3)
        if debug_lane.get("hood_box") is not None:
            X1,Y1,X2,Y2 = debug_lane["hood_box"]
            cv2.rectangle(vis, (X1,Y1), (X2,Y2), (0,0,255), 2)

        # draw TL if any
        if tl_box is not None:
            x1,y1,x2,y2 = map(int, tl_box)
            color = (0,0,255) if tl_state=="red" else (0,255,255) if tl_state=="yellow" else (0,255,0)
            cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)
            cv2.putText(vis, f"TL:{tl_state}", (x1, max(0,y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # draw tracks and compute ADAS metrics
        for tr in tracks:
            try:
                if hasattr(tr, "is_confirmed") and (not tr.is_confirmed()):
                    continue
            except Exception:
                pass
            try:
                tid = int(tr.track_id)
            except Exception:
                tid = abs(hash(str(tr.track_id))) % 10000
            try:
                ltrb = tr.to_ltrb()
            except Exception:
                ltrb = tr.to_tlbr()
            x1,y1,x2,y2 = map(int, ltrb)
            cx = (x1+x2)//2; cy = y2
            color = PALETTE[tid % len(PALETTE)]

            # pixel-speed fallback
            speed_px_s = 0.0
            if tid in last_centroid:
                px,py,t_prev = last_centroid[tid]
                dt = max(1e-6, t_frame - t_prev)
                dist_px = math.hypot(cx - px, cy - py)
                speed_px_s = dist_px / dt
            last_centroid[tid] = (cx,cy,t_frame)

            # TTC fallback
            metric_ttc = (H - y2) / (speed_px_s + 1e-6) if speed_px_s > 1e-6 else float('inf')

            # lane departure
            lane_depart = False
            lateral_px = None
            if lane_center is not None:
                (lx1,ly1),(lx2,ly2) = lane_center
                num = abs((lx2-lx1)*(ly1-cy) - (lx1-cx)*(ly2-ly1))
                den = math.hypot(lx2-lx1, ly2-ly1) + 1e-6
                lateral_px = num/den
                lane_depart = lateral_px > DEPARTURE_THRESHOLD_PX

            # alarm decision
            alarm = (metric_ttc is not None and metric_ttc <= TTC_THRESHOLD)
            if alarm:
                now = time.time()
                last = last_alarm_time.get(tid, -9999)
                if (now - last) >= ALARM_COOLDOWN:
                    last_alarm_time[tid] = now
                    if BEEP_ON_ALARM:
                        try:
                            import winsound
                            winsound.Beep(750, 160)
                        except Exception:
                            pass

            # draw bounding box + label
            box_color = (0,0,255) if alarm or lane_depart else color
            cv2.rectangle(vis, (x1,y1), (x2,y2), box_color, 2)
            label = f"ID:{tid} {0 if speed_px_s is None else int(speed_px_s)}px/s"
            if metric_ttc != float('inf'):
                label += f" TTC:{metric_ttc:.1f}s"
            if lane_depart:
                label += " L-DEP"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            bx1, by1 = x1, max(0, y1 - th - 6)
            bx2, by2 = x1 + tw + 6, by1 + th + 6
            cv2.rectangle(vis, (bx1, by1), (bx2, by2), box_color, -1)
            cv2.putText(vis, label, (x1+3, by2-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            # log event
            events.append({
                "frame": frame_id, "time": t_frame - t_start, "track_id": tid,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "speed_px_s": speed_px_s, "ttc_s": metric_ttc, "lane_dep": int(lane_depart),
                "tl_state": tl_state
            })

        # HUD
        cv2.putText(vis, f"Frame:{frame_id} TL:{tl_state if tl_state else 'none'}", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        writer.write(vis)
        frame_id += 1

    # cleanup
    cap.release()
    writer.release()

    # save CSV
    keys = ["frame","time","track_id","x1","y1","x2","y2","speed_px_s","ttc_s","lane_dep","tl_state"]
    with open(str(out_csv_path), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in events:
            w.writerow({k: r.get(k, None) for k in keys})

    print("DONE. Output video:", output_path, "Events CSV:", out_csv_path)

# -----------------------------
# Argument parsing
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Run ADAS demo on a local video")
    p.add_argument("--video", required=True, help="Input video path")
    p.add_argument("--output", default=str(Path(DEFAULT_OUTPUT_DIR)/"adas_output.mp4"), help="Output annotated video path")
    p.add_argument("--csv", default=str(Path(DEFAULT_OUTPUT_DIR)/"adas_events.csv"), help="Output CSV path")
    p.add_argument("--detect", default=DEFAULT_DETECT_WEIGHTS, help="Detection weights (YOLO)")
    p.add_argument("--seg", default=DEFAULT_SEG_WEIGHTS, help="Segmentation weights (YOLO) or none")
    p.add_argument("--device", default=DEVICE_DEFAULT, help="Device (cpu or cuda)")
    p.add_argument("--conf", type=float, default=DEFAULT_CONF, help="Detection confidence threshold")
    p.add_argument("--tl", action="store_true", help="Enable traffic light color detection")
    return p.parse_args()

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    args = parse_args()
    run(args.video, args.output, args.csv, args.detect, args.seg, args.device, args.conf, args.tl)
