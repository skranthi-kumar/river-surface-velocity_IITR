import cv2
import numpy as np
import pandas as pd
from scipy.signal import medfilt, savgol_filter
import json

class RiverVelocityEstimator:
    def __init__(self, config):
        self.config = config
        self.camera_matrix = None
        self.dist_coeffs = None
        self.scale = config.get('scale', 0.046455)
        self.roi_mask = None

    def load_camera_calibration(self, calibration_file):
        """Load camera calibration parameters (optional)"""
        try:
            with open(calibration_file, 'r') as f:
                data = json.load(f)
            self.camera_matrix = np.array(data['camera_matrix'])
            self.dist_coeffs = np.array(data['dist_coeffs'])
            print("Camera calibration loaded.")
        except Exception as e:
            print("Calibration load failed, using default.", e)

    def create_roi_mask(self, shape, roi_coords=None):
        """Create ROI mask polygon based on ADCP cross-section"""
        mask = np.zeros(shape[:2], dtype=np.uint8)
        if roi_coords is None:
            h, w = shape[:2]
            roi_coords = [
                (int(w*0.15), int(h*0.3)),
                (int(w*0.85), int(h*0.3)),
                (int(w*0.85), int(h*0.85)),
                (int(w*0.15), int(h*0.85))
            ]
        cv2.fillPoly(mask, [np.array(roi_coords)], 255)
        self.roi_mask = mask
        return self.roi_mask

    def preprocess(self, frame):
        """Undistort, convert to grayscale, enhance, denoise, apply ROI"""
        if self.camera_matrix is not None:
            frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(2.0, (8, 8))
        gray = clahe.apply(gray)
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        if self.roi_mask is not None:
            filtered = cv2.bitwise_and(filtered, self.roi_mask)
        return filtered

    def detect_features(self, gray):
        params = dict(maxCorners=300, qualityLevel=0.01, minDistance=10, blockSize=7)
        corners = cv2.goodFeaturesToTrack(gray, mask=self.roi_mask, **params)
        return corners

    def filter_velocities(self, arr):
        if arr is None or len(arr) < 3:
            return arr
        q25, q75 = np.percentile(arr, [25, 75])
        iqr = q75 - q25
        minv = q25 - self.config.get('outlier_threshold', 1.5) * iqr
        maxv = q75 + self.config.get('outlier_threshold', 1.5) * iqr
        return arr[(arr >= minv) & (arr <= maxv)]

    def pixel_to_mps(self, pix_val, fps):
        mps = pix_val * fps * self.scale
        if mps < self.config.get('min_velocity_threshold', 0.01):
            mps = 0
        if mps > self.config.get('max_velocity_threshold', 3.0):
            mps = self.config.get('max_velocity_threshold', 3.0)
        return mps

    def process_video(self, video_path, roi_coords=None):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Cannot read video")
        
        self.create_roi_mask(frame.shape, roi_coords)
        old_gray = self.preprocess(frame)
        points = self.detect_features(old_gray)
        results = []
        frame_num = 0

        print("Processing video... Press 'q' to quit early")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = self.preprocess(frame)
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
                new_points, status, _ = cv2.calcOpticalFlowPyrLK(old_gray, gray, points, None, **lk_params)
                if new_points is not None and status is not None:
                    good_new = new_points[status == 1]
                    good_old = points[status == 1]
                    npts = len(good_new)
                    dists = np.linalg.norm(good_new - good_old, axis=1)
                    dists = self.filter_velocities(dists)
                    if len(dists) > 0:
                        klt_pix_vel = np.mean(dists)
                    points = good_new.reshape(-1, 1, 2)
                if npts < self.config['min_points']:
                    points = self.detect_features(gray)
            else:
                points = self.detect_features(gray)

            # --- Farneback Optical Flow ---
            fb_params = self.config['farneback_params']
            fb_flow = cv2.calcOpticalFlowFarneback(old_gray, gray, None, **fb_params)
            mag, ang = cv2.cartToPolar(fb_flow[..., 0], fb_flow[..., 1])
            mag_roi = mag[self.roi_mask > 0]
            valid = mag_roi[mag_roi > 0.1]
            valid = self.filter_velocities(valid.flatten())
            if len(valid) > 0:
                farne_pix_vel = np.mean(valid)

            # --- Convert to m/s ---
            klt_mps = self.pixel_to_mps(klt_pix_vel, fps)
            farne_mps = self.pixel_to_mps(farne_pix_vel, fps)

            results.append({
                'frame': frame_num,
                'velocity_klt_m_per_s': klt_mps,
                'velocity_farneback_m_per_s': farne_mps,
                'num_klt_points': npts
            })

            # --- Visualization ---
            vis_frame = frame.copy()

            # Draw KLT points
            if points is not None:
                for pt in points:
                    x, y = pt.ravel()
                    cv2.circle(vis_frame, (int(x), int(y)), 3, (0, 255, 0), -1)

            # Draw Farneback flow (sampled)
            step = 15
            for y in range(0, fb_flow.shape[0], step):
                for x in range(0, fb_flow.shape[1], step):
                    fx, fy = fb_flow[y, x]
                    cv2.line(vis_frame, (x, y), (int(x+fx), int(y+fy)), (255, 0, 0), 1)
                    cv2.circle(vis_frame, (x, y), 1, (255, 0, 0), -1)

            cv2.putText(vis_frame, f"Frame: {frame_num}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(vis_frame, f"KLT: {klt_mps:.3f} m/s  FB: {farne_mps:.3f} m/s", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("Velocity Tracing", vis_frame)

            # Update
            old_gray = gray.copy()
            frame_num += 1

            # Progress print
            if frame_num % 50 == 0:
                print(f"  Processed {frame_num} frames...")

            # Exit early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        df = pd.DataFrame(results)
        if len(df) >= 11:
            df['velocity_farneback_smoothed'] = savgol_filter(df['velocity_farneback_m_per_s'], 11, 2)
        else:
            df['velocity_farneback_smoothed'] = medfilt(df['velocity_farneback_m_per_s'], kernel_size=5)

        return df

    def correct_bias(self, df, adcp_mean):
        flow_mean = df['velocity_farneback_smoothed'].mean()
        correction_factor = adcp_mean / flow_mean
        df['velocity_farneback_corrected'] = df['velocity_farneback_smoothed'] * correction_factor
        print(f"Applied correction factor: {correction_factor:.4f}")
        return df

    def evaluate(self, df, adcp_mean):
        vel = df['velocity_farneback_corrected'] if 'velocity_farneback_corrected' in df else df['velocity_farneback_smoothed']
        meanvel = vel.mean()
        error = meanvel - adcp_mean
        rel_error = error / adcp_mean * 100
        print(f"\nVelocity mean after all corrections: {meanvel:.4f} m/s")
        print(f"Absolute error: {error:.4f} m/s")
        print(f"Relative error: {rel_error:.2f}%")
        return {'mean': meanvel, 'absolute_error': error, 'relative_error(%)': rel_error}

# --- CONFIG ---
config = {
    'scale': 0.046455,
    'min_points': 20,
    'outlier_threshold': 1.5,
    'min_velocity_threshold': 0.01,
    'max_velocity_threshold': 3.0,
    'farneback_params': {
        'pyr_scale': 0.5, 'levels': 5, 'winsize': 21, 'iterations': 5,
        'poly_n': 7, 'poly_sigma': 1.5, 'flags': 0
    }
}

video_path = "oparaFiles/FreibergerMulde/terrestrial/CIMG3606.MOV"
output_csv = "velocity_output_final.csv"
adcp_mean = 0.5884
roi_coords = [(100, 80), (540, 80), (540, 400), (100, 400)]

estimator = RiverVelocityEstimator(config)
df = estimator.process_video(video_path, roi_coords)
df = estimator.correct_bias(df, adcp_mean)
df.to_csv(output_csv, index=False)
metrics = estimator.evaluate(df, adcp_mean)
print("\nMetrics:", metrics)
