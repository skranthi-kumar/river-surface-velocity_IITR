import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==== CONFIG ====
video_path = "data/samples/sample_river_1.mp4"   # <-- replace with your file
output_csv = "velocity_output.csv"

# === Load video ===
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
dt = 1.0 / fps

# --- Parameters for ShiTomasi corner detection ---
feature_params = dict(maxCorners=200,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# --- Parameters for Lucas-Kanade optical flow ---
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create random colors for visualization
color = np.random.randint(0, 255, (200, 3))

# === Read first frame ===
ret, old_frame = cap.read()
if not ret:
    print("Error: could not read video")
    exit()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# For visualization
mask = np.zeros_like(old_frame)

# Store results
velocities_klt = []
velocities_farneback = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- 1. KLT Optical Flow ---
    if p0 is not None:
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            # compute displacement and velocity
            displacements = good_new - good_old
            dists = np.linalg.norm(displacements, axis=1)
            vel = np.mean(dists) / dt  # px/sec
            velocities_klt.append(vel)

            # draw tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
                frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
            img = cv2.add(frame, mask)

            cv2.imshow("KLT Optical Flow", img)

            # update points
            p0 = good_new.reshape(-1, 1, 2)
        else:
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)

    # --- 2. Farneback Dense Optical Flow ---
    flow = cv2.calcOpticalFlowFarneback(old_gray, frame_gray, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mean_vel = np.mean(mag) / dt  # px/sec
    velocities_farneback.append(mean_vel)

    hsv = np.zeros_like(frame)
    hsv[..., 1] = 255
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow("Farneback Optical Flow", rgb)

    # --- Update ---
    old_gray = frame_gray.copy()

    # Exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# === Save results ===
import pandas as pd
df = pd.DataFrame({
    "frame": np.arange(len(velocities_klt)),
    "velocity_klt_px_per_s": velocities_klt,
    "velocity_farneback_px_per_s": velocities_farneback[:len(velocities_klt)]
})
df.to_csv(output_csv, index=False)

print("Saved results to", output_csv)

# === Plot ===
plt.plot(df["frame"], df["velocity_klt_px_per_s"], label="KLT")
plt.plot(df["frame"], df["velocity_farneback_px_per_s"], label="Farneback")
plt.xlabel("Frame")
plt.ylabel("Velocity (px/s)")
plt.legend()
plt.show()
