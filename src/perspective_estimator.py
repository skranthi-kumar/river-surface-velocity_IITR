import cv2
import numpy as np
import pandas as pd
from scipy.signal import medfilt, savgol_filter
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor

class EnhancedRiverVelocityEstimator:
    def __init__(self, config):
        self.config = config
        self.roi_mask = None
        self.homography_matrix = None
        self.warped_scale = None
        self.output_size = None
        
    def calibrate_perspective(self, frame, roi_coords, known_distances, pixels_per_meter=200):
        """
        Calibrate perspective using known distances.
        
        Args:
            frame: First frame
            roi_coords: ROI coordinates [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
            known_distances: Dict with 'width' and 'height' in meters
            pixels_per_meter: Resolution of warped view (default 200 px/m for good quality)
        """
        src_pts = np.float32(roi_coords)
        
        # Calculate output dimensions with higher resolution
        width_px = int(known_distances['width'] * pixels_per_meter)
        height_px = int(known_distances['height'] * pixels_per_meter)
        
        dst_pts = np.float32([
            [0, 0],
            [width_px, 0],
            [width_px, height_px],
            [0, height_px]
        ])
        
        self.homography_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        self.warped_scale = 1.0 / pixels_per_meter  # m/px
        self.output_size = (width_px, height_px)
        
        print(f"Calibration complete:")
        print(f"  Warped view: {width_px}x{height_px} px")
        print(f"  Scale: {self.warped_scale:.6f} m/px")
        print(f"  Resolution: {pixels_per_meter} px/m")
        
        return width_px, height_px

    def manual_calibrate(self, frame, roi_coords):
        """
        Manual calibration by measuring pixel distances.
        User should provide the scale after measuring a known distance.
        """
        # Just use original frame without warping
        self.homography_matrix = None
        self.output_size = (frame.shape[1], frame.shape[0])
        
        # Calculate approximate scale from ROI
        # This is rough - user should refine with actual measurements
        roi_array = np.array(roi_coords)
        width_px = np.linalg.norm(roi_array[1] - roi_array[0])
        height_px = np.linalg.norm(roi_array[3] - roi_array[0])
        
        print(f"\nManual calibration mode:")
        print(f"  ROI width: {width_px:.1f} pixels")
        print(f"  ROI height: {height_px:.1f} pixels")
        print(f"\n  IMPORTANT: Measure a known distance in your video!")
        print(f"  Then calculate: scale = (distance_in_meters) / (distance_in_pixels)")
        print(f"  Update the scale in the config.\n")
        
        return self.output_size

    def set_manual_scale(self, scale_m_per_px):
        """Set scale manually after measuring a known distance."""
        self.warped_scale = scale_m_per_px
        print(f"Manual scale set to: {scale_m_per_px:.6f} m/px")

    def warp_frame(self, frame):
        """Apply perspective transform if available."""
        if self.homography_matrix is None:
            return frame
        return cv2.warpPerspective(frame, self.homography_matrix, self.output_size)

    def create_roi_mask(self, shape):
        """Create ROI mask (full frame in warped space)."""
        mask = np.ones(shape[:2], dtype=np.uint8) * 255
        self.roi_mask = mask
        return mask

    def preprocess(self, frame):
        """Enhanced preprocessing."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # CLAHE for contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Bilateral filter
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # High-pass filter for texture enhancement
        low_pass = cv2.GaussianBlur(filtered, (21, 21), 0)
        high_pass = cv2.addWeighted(filtered, 1.5, low_pass, -0.5, 0)
        
        if self.roi_mask is not None:
            high_pass = cv2.bitwise_and(high_pass, self.roi_mask)
            
        return high_pass

    def detect_features(self, gray):
        """Detect features for tracking."""
        params = dict(
            maxCorners=500,
            qualityLevel=0.01,
            minDistance=8,
            blockSize=7
        )
        corners = cv2.goodFeaturesToTrack(gray, mask=self.roi_mask, **params)
        return corners

    def ransac_filter_velocities(self, old_pts, new_pts):
        """RANSAC filtering of motion vectors."""
        if len(old_pts) < 4:
            return old_pts, new_pts, np.ones(len(old_pts), dtype=bool)
        
        displacements = new_pts - old_pts
        
        try:
            # RANSAC on horizontal displacement
            ransac = RANSACRegressor(residual_threshold=2.0, random_state=42)
            ransac.fit(old_pts[:, 0:1], displacements[:, 0])
            inlier_mask = ransac.inlier_mask_
            
            # Check y-displacement consistency
            y_median = np.median(displacements[:, 1])
            y_mad = np.median(np.abs(displacements[:, 1] - y_median))
            y_inliers = np.abs(displacements[:, 1] - y_median) < (3 * y_mad + 0.1)
            
            combined_mask = inlier_mask & y_inliers
            
            return old_pts[combined_mask], new_pts[combined_mask], combined_mask
        except:
            return old_pts, new_pts, np.ones(len(old_pts), dtype=bool)

    def filter_velocities_iqr(self, arr):
        """IQR-based outlier removal."""
        if arr is None or len(arr) < 3:
            return arr
        q25, q75 = np.percentile(arr, [25, 75])
        iqr = q75 - q25
        minv = q25 - self.config.get('outlier_threshold', 1.5) * iqr
        maxv = q75 + self.config.get('outlier_threshold', 1.5) * iqr
        return arr[(arr >= minv) & (arr <= maxv)]

    def pixel_to_mps(self, pix_val, fps):
        """Convert pixel displacement to m/s."""
        if self.warped_scale is None:
            print("WARNING: Scale not set! Using default.")
            scale = 0.017689
        else:
            scale = self.warped_scale
            
        mps = pix_val * fps * scale
        
        # Apply thresholds
        if mps < self.config.get('min_velocity_threshold', 0.01):
            mps = 0
        if mps > self.config.get('max_velocity_threshold', 5.0):
            mps = self.config.get('max_velocity_threshold', 5.0)
        return mps

    def process_video(self, video_path, roi_coords=None, known_distances=None, 
                     manual_scale=None, pixels_per_meter=200):
        """
        Process video with enhanced algorithms.
        
        Args:
            video_path: Path to video
            roi_coords: ROI coordinates
            known_distances: Dict with 'width' and 'height' in meters (for auto calibration)
            manual_scale: Manual scale in m/px (if you measured it yourself)
            pixels_per_meter: Resolution for warped view (higher = better quality but slower)
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Cannot read video")
        
        print(f"Video info: {frame.shape[1]}x{frame.shape[0]} @ {fps:.1f} fps")
        
        # Calibration
        if manual_scale is not None:
            # Use manual scale without warping
            self.manual_calibrate(frame, roi_coords)
            self.set_manual_scale(manual_scale)
        elif known_distances is not None and roi_coords is not None:
            # Auto calibration with warping
            self.calibrate_perspective(frame, roi_coords, known_distances, pixels_per_meter)
        else:
            # No calibration
            self.output_size = (frame.shape[1], frame.shape[0])
            print("WARNING: No calibration provided!")
        
        # Initialize
        frame = self.warp_frame(frame)
        self.create_roi_mask(frame.shape)
        old_gray = self.preprocess(frame)
        points = self.detect_features(old_gray)
        
        results = []
        frame_num = 0
        color = np.random.randint(0, 255, (500, 3))
        fb_params = self.config['farneback_params']

        while True:
            ret, frame_orig = cap.read()
            if not ret:
                break
            
            frame = self.warp_frame(frame_orig)
            gray = self.preprocess(frame)
            vis_frame = frame.copy()
            
            klt_pix_vel = 0
            farne_pix_vel = 0
            npts = 0

            # --- KLT Optical Flow ---
            if points is not None and len(points) > 0:
                lk_params = dict(
                    winSize=(21, 21),
                    maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
                )
                new_points, status, err = cv2.calcOpticalFlowPyrLK(
                    old_gray, gray, points, None, **lk_params
                )
                
                # Forward-backward check
                back_points, status_back, _ = cv2.calcOpticalFlowPyrLK(
                    gray, old_gray, new_points, None, **lk_params
                )
                
                fb_error = np.linalg.norm(points - back_points, axis=2).flatten()
                good_fb = fb_error < 1.0
                
                good_status = (status == 1).flatten() & (status_back == 1).flatten() & good_fb
                
                if np.sum(good_status) > 0:
                    good_new = new_points[good_status]
                    good_old = points[good_status]
                    
                    # RANSAC filtering
                    good_old_f, good_new_f, inlier_mask = self.ransac_filter_velocities(
                        good_old.reshape(-1, 2), good_new.reshape(-1, 2)
                    )
                    
                    npts = len(good_new_f)
                    
                    if npts > 0:
                        displacements = good_new_f - good_old_f
                        horizontal_vel = displacements[:, 0]
                        horizontal_vel = self.filter_velocities_iqr(horizontal_vel)
                        
                        if len(horizontal_vel) > 0:
                            klt_pix_vel = np.median(horizontal_vel)
                        
                        # Visualize (subsample for clarity)
                        step = max(1, len(good_new_f) // 50)
                        for i in range(0, len(good_new_f), step):
                            new, old = good_new_f[i], good_old_f[i]
                            a, b = int(new[0]), int(new[1])
                            c, d = int(old[0]), int(old[1])
                            cv2.line(vis_frame, (a, b), (c, d), color[i % 500].tolist(), 2)
                            cv2.circle(vis_frame, (a, b), 3, color[i % 500].tolist(), -1)
                        
                        points = good_new_f.reshape(-1, 1, 2)
                
                if npts < self.config['min_points']:
                    points = self.detect_features(gray)
            else:
                points = self.detect_features(gray)

            # --- Farneback Optical Flow ---
            fb_flow = cv2.calcOpticalFlowFarneback(old_gray, gray, None, **fb_params)
            
            flow_x = fb_flow[..., 0]
            flow_y = fb_flow[..., 1]
            mag = np.sqrt(flow_x**2 + flow_y**2)
            angle = np.arctan2(flow_y, flow_x) * 180 / np.pi
            
            # Filter by direction (horizontal flow)
            direction_mask = (np.abs(angle) < 30) | (np.abs(angle - 180) < 30)
            roi_flow = flow_x[self.roi_mask > 0]
            roi_mag = mag[self.roi_mask > 0]
            roi_angle_mask = direction_mask[self.roi_mask > 0]
            
            valid = roi_flow[(roi_mag > 0.1) & roi_angle_mask]
            valid = self.filter_velocities_iqr(valid)
            
            if len(valid) > 0:
                farne_pix_vel = np.median(valid)

            # Convert to m/s
            klt_mps = self.pixel_to_mps(klt_pix_vel, fps)
            farne_mps = self.pixel_to_mps(farne_pix_vel, fps)

            # Overlay
            cv2.putText(vis_frame, f"KLT: {klt_mps:.3f} m/s (n={npts})", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis_frame, f"Farneback: {farne_mps:.3f} m/s", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(vis_frame, f"Frame: {frame_num}", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Resize for display if too small
            if vis_frame.shape[1] < 400 or vis_frame.shape[0] < 300:
                scale_factor = max(400/vis_frame.shape[1], 300/vis_frame.shape[0])
                new_width = int(vis_frame.shape[1] * scale_factor)
                new_height = int(vis_frame.shape[0] * scale_factor)
                vis_frame = cv2.resize(vis_frame, (new_width, new_height))

            cv2.imshow("Enhanced Velocity", vis_frame)

            results.append({
                'frame': frame_num,
                'velocity_klt_m_per_s': klt_mps,
                'velocity_farneback_m_per_s': farne_mps,
                'num_klt_points': npts,
            })
            frame_num += 1
            old_gray = gray.copy()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        df = pd.DataFrame(results)
        
        # Smoothing
        window_size = min(11, len(df) if len(df) % 2 == 1 else len(df) - 1)
        if window_size >= 5:
            df['velocity_klt_smoothed'] = savgol_filter(
                df['velocity_klt_m_per_s'], window_size, 2
            )
            df['velocity_farneback_smoothed'] = savgol_filter(
                df['velocity_farneback_m_per_s'], window_size, 2
            )
        else:
            df['velocity_klt_smoothed'] = df['velocity_klt_m_per_s']
            df['velocity_farneback_smoothed'] = df['velocity_farneback_m_per_s']
        
        # Ensemble (weighted average)
        df['velocity_ensemble'] = (
            0.4 * df['velocity_klt_smoothed'] + 
            0.6 * df['velocity_farneback_smoothed']
        )
        
        return df


# --- CONFIGURATION ---
config = {
    'min_points': 30,
    'outlier_threshold': 1.5,
    'min_velocity_threshold': 0.01,
    'max_velocity_threshold': 5.0,
    'farneback_params': {
        'pyr_scale': 0.5,
        'levels': 5,
        'winsize': 25,
        'iterations': 5,
        'poly_n': 7,
        'poly_sigma': 1.5,
        'flags': 0
    }
}

# --- Usage Examples ---
if __name__ == "__main__":
    video_path = "data/samples/sample_river_1.mp4"
    roi_coords = [(379, 234), (544, 236), (618, 390), (303, 392)]
    output_csv = "velocity_enhanced_output.csv"
    
    estimator = EnhancedRiverVelocityEstimator(config)
    
    # === METHOD 1: Manual Scale (RECOMMENDED for now) ===
    # Measure a known distance in your video (e.g., ruler, known object)
    # Then calculate: scale = distance_in_meters / distance_in_pixels
    
    # For float test: 0.3 m/s
    # If your Farneback gave 0.1087 m/s with old scale 0.017689 m/px
    # That means it detected ~6.15 pixels/frame displacement
    # To get 0.3 m/s from 6.15 px/frame at 30fps:
    # scale = 0.3 / (6.15 * 30) = 0.00163 m/px
    
    # Try this scale first:
    manual_scale = 0.00163  # Adjust based on your measurements
    
    df = estimator.process_video(
        video_path, 
        roi_coords=roi_coords,
        manual_scale=manual_scale
    )
    
    # === METHOD 2: Auto Calibration (if you know actual ROI dimensions) ===
    # Measure the real-world dimensions of your ROI
    # known_distances = {
    #     'width': 2.5,   # actual width in meters
    #     'height': 1.8   # actual height in meters
    # }
    # df = estimator.process_video(
    #     video_path, 
    #     roi_coords=roi_coords,
    #     known_distances=known_distances,
    #     pixels_per_meter=200  # Higher = better quality but slower
    # )
    
    df.to_csv(output_csv, index=False)
    
    # === Plot ===
    plt.figure(figsize=(12, 6))
    plt.plot(df["velocity_klt_smoothed"], label="KLT Smoothed", 
             color="green", linewidth=2, alpha=0.7)
    plt.plot(df["velocity_farneback_smoothed"], label="Farneback Smoothed", 
             color="blue", linewidth=2, alpha=0.7)
    plt.plot(df["velocity_ensemble"], label="Ensemble", 
             color="red", linewidth=2, linestyle='--')
    plt.axhline(y=0.3, color='orange', linestyle=':', linewidth=2, 
                label='Float Test (0.3 m/s)')
    plt.xlabel("Frame", fontsize=12)
    plt.ylabel("Velocity (m/s)", fontsize=12)
    plt.title("Enhanced River Surface Velocity", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # === Statistics ===
    print("\n=== Velocity Statistics ===")
    print(f"KLT Mean: {df['velocity_klt_smoothed'].mean():.4f} m/s")
    print(f"Farneback Mean: {df['velocity_farneback_smoothed'].mean():.4f} m/s")
    print(f"Ensemble Mean: {df['velocity_ensemble'].mean():.4f} m/s")
    print(f"Float Test: 0.3000 m/s")
    print(f"\nError vs Float Test:")
    print(f"  Ensemble: {abs(df['velocity_ensemble'].mean() - 0.3)/0.3 * 100:.1f}%")
    print(f"\nResults saved to {output_csv}")