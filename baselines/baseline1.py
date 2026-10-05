import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ==== CONFIG ====
video_path = "data/samples/sample_river_1.mp4"
output_csv = "velocity_output_test.csv"

# Tracking parameters
MIN_POINTS = 15  # Minimum points before re-detection
TRAIL_RESET_FRAMES = 30  # Clear trails every N frames

# ==== CALIBRATED SCALE (from GCP analysis) ====
scale = 0.046455  # m/px (median scale from Ground Control Points)
print(f"Using calibrated scale: {scale:.6f} m/px")
print(f"Ground Sample Distance (GSD): {scale*100:.2f} cm/pixel")
print("="*70)

# === Load video ===
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: could not open video")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
dt = 1.0 / fps if fps > 0 else 1/30.0  # fallback to 30fps

print(f"Video FPS: {fps:.2f}, Frame interval: {dt:.4f}s")
print("="*70)

# --- ShiTomasi corner detection ---
feature_params = dict(
    maxCorners=200,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7
)

# --- Lucas-Kanade optical flow ---
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

# Create random colors
color = np.random.randint(0, 255, (200, 3))

# === Read first frame ===
ret, old_frame = cap.read()
if not ret:
    print("Error: could not read first frame")
    exit()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Visualization mask
mask = np.zeros_like(old_frame)

# Store results
results = []
frame_num = 0

print("Processing video... Press 'q' to quit early\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Initialize velocity values
    vel_klt_px = 0.0
    vel_klt_mps = 0.0
    vel_farneback_px = 0.0
    vel_farneback_mps = 0.0
    num_points = 0

    # --- 1. KLT Optical Flow ---
    if p0 is not None and len(p0) > 0:
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        
        if p1 is not None and st is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            num_points = len(good_new)

            if num_points > 0:
                # Compute displacement and velocity
                displacements = good_new - good_old
                dists = np.linalg.norm(displacements, axis=1)
                vel_klt_px = np.mean(dists) / dt  # px/sec
                vel_klt_mps = vel_klt_px * scale  # m/s
                
                # Draw tracks
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = int(new[0]), int(new[1])
                    c, d = int(old[0]), int(old[1])
                    mask = cv2.line(mask, (a, b), (c, d), color[i % len(color)].tolist(), 2)
                    frame = cv2.circle(frame, (a, b), 5, color[i % len(color)].tolist(), -1)
                
                # Update points
                p0 = good_new.reshape(-1, 1, 2)
            
            # Re-detect if too few points
            if num_points < MIN_POINTS:
                p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
                mask = np.zeros_like(old_frame)  # Clear trails
        else:
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
    else:
        p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)

    # Periodically clear trails for cleaner visualization
    if frame_num % TRAIL_RESET_FRAMES == 0 and frame_num > 0:
        mask = np.zeros_like(old_frame)

    # Display KLT
    img_klt = cv2.add(frame, mask)
    cv2.putText(img_klt, f"Frame: {frame_num}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(img_klt, f"Points: {num_points}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(img_klt, f"Vel: {vel_klt_mps:.3f} m/s", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("KLT Optical Flow", img_klt)

    # --- 2. Farneback Dense Optical Flow ---
    flow = cv2.calcOpticalFlowFarneback(
        old_gray, frame_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    vel_farneback_px = np.mean(mag) / dt  # px/sec
    vel_farneback_mps = vel_farneback_px * scale  # m/s

    # Visualize Farneback
    hsv = np.zeros_like(frame)
    hsv[..., 1] = 255
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    cv2.putText(rgb, f"Frame: {frame_num}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(rgb, f"Vel: {vel_farneback_mps:.3f} m/s", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Farneback Optical Flow", rgb)

    # Store results
    results.append({
        "frame": frame_num,
        "time_s": frame_num / fps,
        "velocity_klt_px_per_s": vel_klt_px,
        "velocity_klt_m_per_s": vel_klt_mps,
        "velocity_farneback_px_per_s": vel_farneback_px,
        "velocity_farneback_m_per_s": vel_farneback_mps,
        "num_klt_points": num_points
    })

    # Update for next iteration
    old_gray = frame_gray.copy()
    frame_num += 1

    # Progress indicator
    if frame_num % 50 == 0:
        print(f"  Processed {frame_num} frames...")

    # Exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\nInterrupted by user")
        break

cap.release()
cv2.destroyAllWindows()

print(f"\nProcessed {frame_num} frames")

# === Save results ===
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f"✓ Saved results to {output_csv}")

# === Statistics ===
print("\n" + "="*70)
print("VELOCITY STATISTICS (Pixels/sec)")
print("="*70)
print(f"KLT Optical Flow:")
print(f"  Mean:   {df['velocity_klt_px_per_s'].mean():>8.2f} px/s")
print(f"  Median: {df['velocity_klt_px_per_s'].median():>8.2f} px/s")
print(f"  Std:    {df['velocity_klt_px_per_s'].std():>8.2f} px/s")
print(f"  Min:    {df['velocity_klt_px_per_s'].min():>8.2f} px/s")
print(f"  Max:    {df['velocity_klt_px_per_s'].max():>8.2f} px/s")

print(f"\nFarneback Optical Flow:")
print(f"  Mean:   {df['velocity_farneback_px_per_s'].mean():>8.2f} px/s")
print(f"  Median: {df['velocity_farneback_px_per_s'].median():>8.2f} px/s")
print(f"  Std:    {df['velocity_farneback_px_per_s'].std():>8.2f} px/s")
print(f"  Min:    {df['velocity_farneback_px_per_s'].min():>8.2f} px/s")
print(f"  Max:    {df['velocity_farneback_px_per_s'].max():>8.2f} px/s")

print("\n" + "="*70)
print("VELOCITY STATISTICS (Meters/sec) - CALIBRATED")
print("="*70)
print(f"KLT Optical Flow:")
print(f"  Mean:   {df['velocity_klt_m_per_s'].mean():>8.4f} m/s")
print(f"  Median: {df['velocity_klt_m_per_s'].median():>8.4f} m/s")
print(f"  Std:    {df['velocity_klt_m_per_s'].std():>8.4f} m/s")
print(f"  Min:    {df['velocity_klt_m_per_s'].min():>8.4f} m/s")
print(f"  Max:    {df['velocity_klt_m_per_s'].max():>8.4f} m/s")

print(f"\nFarneback Optical Flow:")
print(f"  Mean:   {df['velocity_farneback_m_per_s'].mean():>8.4f} m/s")
print(f"  Median: {df['velocity_farneback_m_per_s'].median():>8.4f} m/s")
print(f"  Std:    {df['velocity_farneback_m_per_s'].std():>8.4f} m/s")
print(f"  Min:    {df['velocity_farneback_m_per_s'].min():>8.4f} m/s")
print(f"  Max:    {df['velocity_farneback_m_per_s'].max():>8.4f} m/s")

print(f"\nTracking Statistics:")
print(f"  Avg tracked points: {df['num_klt_points'].mean():.1f}")
print(f"  Min tracked points: {df['num_klt_points'].min()}")
print(f"  Max tracked points: {df['num_klt_points'].max()}")

print("="*70)

# === Plot ===
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

# Velocity comparison (m/s)
ax1.plot(df["frame"], df["velocity_klt_m_per_s"], label="KLT", alpha=0.7, linewidth=1.5, color='blue')
ax1.plot(df["frame"], df["velocity_farneback_m_per_s"], label="Farneback", alpha=0.7, linewidth=1.5, color='red')
ax1.axhline(y=df["velocity_klt_m_per_s"].mean(), color='blue', linestyle='--', alpha=0.5, label=f'KLT Mean: {df["velocity_klt_m_per_s"].mean():.3f} m/s')
ax1.axhline(y=df["velocity_farneback_m_per_s"].mean(), color='red', linestyle='--', alpha=0.5, label=f'Farneback Mean: {df["velocity_farneback_m_per_s"].mean():.3f} m/s')
ax1.set_ylabel("Velocity (m/s)", fontsize=11)
ax1.legend(loc='best', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_title(f"Surface Velocity - Optical Flow Comparison (Scale: {scale:.6f} m/px)", fontsize=12, fontweight='bold')

# Number of tracked points
ax2.plot(df["frame"], df["num_klt_points"], color='green', linewidth=1.5)
ax2.axhline(y=df["num_klt_points"].mean(), color='green', linestyle='--', alpha=0.5, label=f'Mean: {df["num_klt_points"].mean():.1f}')
ax2.set_xlabel("Frame Number", fontsize=11)
ax2.set_ylabel("KLT Points Tracked", fontsize=11)
ax2.legend(loc='best', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_title("Number of Tracked Feature Points", fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig("velocity_analysis.png", dpi=150, bbox_inches='tight')
print(f"\n✓ Saved plot to velocity_analysis.png")
plt.show()

print("\nAnalysis complete!")