# --------------------------------------------------------------
# universal_roi_calibrator.py
#   • Auto-detects flowing-water ROI
#   • Falls back to manual click if you press 'm'
#   • Uses camera geometry for scale
#   • Prints ready-to-paste values
# --------------------------------------------------------------

import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple

# --------------------------------------------------------------
# CONFIGURATION (tweak only if you know your camera)
# --------------------------------------------------------------
VIDEO_PATH      = "data/samples/sample_river_1.mp4"
CAMERA_HEIGHT   = 10.0      # meters
CAMERA_ANGLE    = 45.0      # degrees (downward tilt)
ASSUMED_VELOCITY = 1.0     # m/s – only used as a sanity check

# --------------------------------------------------------------
# GLOBAL STATE
# --------------------------------------------------------------
manual_points: List[Tuple[int, int]] = []
auto_roi: List[Tuple[int, int]] = None
frame_copy = None
frame_orig = None

# --------------------------------------------------------------
# 1. AUTOMATIC ROI DETECTION
# --------------------------------------------------------------
def auto_detect_roi(frame: np.ndarray, video_path: str) -> List[Tuple[int, int]]:
    h, w = frame.shape[:2]

    # ---- 1. Water mask (HSV) ---------------------------------
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([80, 30, 30])
    upper = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No water found – using full frame")
        return [(0,0), (w,0), (w,h), (0,h)]

    largest = max(contours, key=cv2.contourArea)
    x, y, ww, hh = cv2.boundingRect(largest)
    water_area = ww * hh
    print(f"Water area: {water_area:,} px")

    # ---- 2. Refine with a few frames of optical flow ----------
    cap = cv2.VideoCapture(video_path)
    frames = []
    for _ in range(20):
        ret, f = cap.read()
        if not ret: break
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
    cap.release()

    if len(frames) < 2:
        print("Not enough frames – using water ROI")
        return [(x,y), (x+ww,y), (x+ww,y+hh), (x,y+hh)]

    # mask the water region
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [np.array([(x,y),(x+ww,y),(x+ww,y+hh),(x,y+hh)])], 255)

    flow_sum = np.zeros_like(frames[0], dtype=np.float32)
    for i in range(len(frames)-1):
        flow = cv2.calcOpticalFlowFarneback(frames[i], frames[i+1], None,
                                            pyr_scale=0.5, levels=3,
                                            winsize=15, iterations=3,
                                            poly_n=5, poly_sigma=1.2, flags=0)
        mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
        flow_sum += mag

    flow_avg = flow_sum / (len(frames)-1)
    flow_in_water = cv2.bitwise_and(flow_avg, flow_avg, mask=roi_mask)

    valid = flow_in_water[roi_mask > 0]
    if len(valid) == 0 or valid.max() == 0:
        print("No flow – using water ROI")
        return [(x,y), (x+ww,y), (x+ww,y+hh), (x,y+hh)]

    thresh = np.percentile(valid[valid > 0], 25)
    high = (flow_in_water > thresh).astype(np.uint8) * 255
    high = cv2.morphologyEx(high, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11)))
    high = cv2.morphologyEx(high, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9)))

    contours, _ = cv2.findContours(high, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No high-flow contours – using water ROI")
        return [(x,y), (x+ww,y), (x+ww,y+hh), (x,y+hh)]

    best = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(best)
    if area < h*w*0.015:          # at least 1.5 % of frame
        print(f"High-flow ROI too small ({area:.0f}px) – using water ROI")
        return [(x,y), (x+ww,y), (x+ww,y+hh), (x,y+hh)]

    rect = cv2.minAreaRect(best)
    box = cv2.boxPoints(rect)
    roi = np.intp(box).tolist()
    print(f"Auto ROI area: {area:.0f} px")
    return roi


# --------------------------------------------------------------
# 2. MANUAL ROI CLICKER (press 'm' to switch)
# --------------------------------------------------------------
def click_callback(event, x, y, flags, param):
    global manual_points, frame_copy
    if event == cv2.EVENT_LBUTTONDOWN:
        manual_points.append((x, y))
        cv2.circle(frame_copy, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Click 4 corners – 'r' reset, 'q' done", frame_copy)

def get_manual_roi(frame: np.ndarray) -> List[Tuple[int, int]]:
    global manual_points, frame_copy
    manual_points = []
    frame_copy = frame.copy()
    cv2.namedWindow("Click 4 corners – 'r' reset, 'q' done")
    cv2.setMouseCallback("Click 4 corners – 'r' reset, 'q' done", click_callback)

    print("\nClick **four corners** of the river ROI (clockwise or counter-clockwise).")
    print("   'r' – reset points")
    print("   'q' – finish")
    while True:
        cv2.imshow("Click 4 corners – 'r' reset, 'q' done", frame_copy)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('r'):
            manual_points = []
            frame_copy = frame.copy()
        elif k == ord('q') and len(manual_points) == 4:
            break

    cv2.destroyAllWindows()
    print(f"Manual ROI: {manual_points}")
    return manual_points


# --------------------------------------------------------------
# 3. SCALE FROM CAMERA GEOMETRY
# --------------------------------------------------------------
def scale_from_camera(roi: List[Tuple[int, int]], frame_h: int, frame_w: int) -> float:
    angle_rad = np.deg2rad(CAMERA_ANGLE)
    ground_dist = CAMERA_HEIGHT / np.tan(angle_rad)
    roi_y = np.mean([p[1] for p in roi])
    perspective = 1.0 + (1.0 - roi_y / frame_h) * 0.7
    scale = (ground_dist / frame_w) * perspective
    return scale


# --------------------------------------------------------------
# 4. MAIN PIPELINE
# --------------------------------------------------------------
def main():
    global auto_roi, frame_orig

    # ---- read first frame -------------------------------------------------
    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame_orig = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Cannot read video")
    h, w = frame_orig.shape[:2]

    # ---- 1. AUTO ROI ------------------------------------------------------
    print("\n" + "="*70)
    print("STEP 1 – AUTO ROI")
    print("="*70)
    auto_roi = auto_detect_roi(frame_orig, VIDEO_PATH)

    # show auto result
    vis = frame_orig.copy()
    cv2.polylines(vis, [np.array(auto_roi)], True, (0,255,0), 3)
    cv2.putText(vis, "AUTO ROI – press any key", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Auto ROI (green) – any key to continue", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # ---- 2. OPTIONAL MANUAL OVERRIDE --------------------------------------
    print("\nPress 'm' for **manual** ROI, any other key to keep auto.")
    key = input(">>> ").strip().lower()
    if key == 'm':
        print("Switching to manual mode...")
        final_roi = get_manual_roi(frame_orig)
    else:
        final_roi = auto_roi
        print("Keeping auto ROI")

    # ---- 3. SCALE ---------------------------------------------------------
    print("\n" + "="*70)
    print("STEP 2 – SCALE (camera geometry)")
    print("="*70)
    scale = scale_from_camera(final_roi, h, w)
    print(f"scale = {scale:.10f} m/px")

    # ---- 4. SAVE & PRINT --------------------------------------------------
    cfg = {
        "scale": scale,
        "roi_coords": final_roi,
        "video": VIDEO_PATH,
        "camera_height": CAMERA_HEIGHT,
        "camera_angle": CAMERA_ANGLE
    }
    with open("universal_calibration.json", "w") as f:
        json.dump(cfg, f, indent=4)
    print("\nSaved → universal_calibration.json")

    print("\n" + "="*70)
    print("COPY-PASTE INTO YOUR VELOCITY CODE")
    print("="*70)
    print(f"scale      = {scale}")
    print(f"roi_coords = {final_roi}")
    print("="*70)

    # optional visualisation
    final_vis = frame_orig.copy()
    cv2.polylines(final_vis, [np.array(final_roi)], True, (0,255,0), 3)
    cv2.putText(final_vis, f"scale={scale:.6f}", (10, h-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.imwrite("final_roi.png", final_vis)
    print("Saved visualisation → final_roi.png")

if __name__ == "__main__":
    main()