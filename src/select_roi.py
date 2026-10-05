import cv2

# --- Configuration ---
import argparse
_p = argparse.ArgumentParser(description="Click 4 ROI corners (clockwise) on the first frame; prints coordinates.")
_p.add_argument("video"); video_path = _p.parse_args().video
points = []

# Mouse callback to capture points
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point clicked: ({x}, {y})")
        # Draw a small circle for visual feedback
        cv2.circle(frame_copy, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Select ROI - click 4 points clockwise", frame_copy)

# Capture the first frame
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Cannot read video")

frame_copy = frame.copy()
cv2.imshow("Select ROI - click 4 points clockwise", frame_copy)
cv2.setMouseCallback("Select ROI - click 4 points clockwise", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(points) != 4:
    raise ValueError(f"Exactly 4 points are required. You clicked {len(points)}")

print("\nROI coordinates (use in your algorithm):")
print(points)
