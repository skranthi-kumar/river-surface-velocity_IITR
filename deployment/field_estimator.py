import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, medfilt
from scipy import stats
import matplotlib.pyplot as plt
from collections import deque

class RiverVelocityEstimator:
    def __init__(self, config):
        self.config = config
        self.scale = config.get('scale', 0.000708)  # m/px, assumed accurate or estimated
        self.scale_uncertainty = config.get('scale_uncertainty', 0.00005)
        self.roi_mask = None
        self.prev_flow = None  # For temporal smoothing

    def estimate_scale_factor(self, frame, reference_points=None, reference_length_m=None):
        """Estimate scale factor from known physical reference in the scene."""
        if reference_points is None or reference_length_m is None:
            print("Warning: No reference points or length provided. Using default scale.")
            return self.scale
        
        # Calculate pixel distance between reference points
        (x1, y1), (x2, y2) = reference_points
        pixel_distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Compute scale (m/px)
        if pixel_distance > 0:
            scale = reference_length_m / pixel_distance
            print(f"Estimated scale factor: {scale:.6f} m/px")
            return scale
        else:
            print("Error: Invalid reference points. Using default scale.")
            return self.scale

    def create_roi_mask(self, shape, roi_coords=None):
        """Create a binary mask for the region of interest."""
        mask = np.zeros(shape[:2], dtype=np.uint8)
        if roi_coords is None:
            h, w = shape[:2]
            roi_coords = [
                (int(w * 0.15), int(h * 0.3)),
                (int(w * 0.85), int(h * 0.3)),
                (int(w * 0.85), int(h * 0.85)),
                (int(w * 0.15), int(h * 0.85))
            ]
        cv2.fillPoly(mask, [np.array(roi_coords, dtype=np.int32)], 255)
        self.roi_mask = mask
        return mask

    def preprocess(self, frame, enhance=True):
        """Enhanced preprocessing with adaptive histogram equalization."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if enhance:
            mean_intensity = np.mean(gray)
            clip_limit = 3.0 if mean_intensity < 100 else 2.0
            clahe = cv2.createCLAHE(clipLimit=clip_limit,
                                   tileGridSize=self.config.get('clahe_grid', (8, 8)))
            gray = clahe.apply(gray)
            gray = cv2.bilateralFilter(gray, d=self.config.get('bilateral_d', 9),
                                      sigmaColor=75, sigmaSpace=75)
            if self.config.get('enable_sharpening', True):
                kernel = np.array([[-0.5, -0.5, -0.5],
                                  [-0.5, 5.0, -0.5],
                                  [-0.5, -0.5, -0.5]])
                gray = cv2.filter2D(gray, -1, kernel * 0.2)
        
        if self.roi_mask is not None:
            gray = cv2.bitwise_and(gray, gray, mask=self.roi_mask)
        
        return gray

    def perspective_warp(self, frame, src_pts, dst_size):
        """Apply perspective correction to frame."""
        if src_pts is None:
            return frame, np.eye(3)
        h, w = dst_size
        dst_pts = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(np.float32(src_pts), dst_pts)
        warped = cv2.warpPerspective(frame, M, dst_size)
        return warped, M

    def detect_features_adaptive(self, gray):
        """Adaptive feature detection with quality assessment and fallback to ORB."""
        params = self.config.get('klt_params', {
            'maxCorners': 500, 'qualityLevel': 0.005, 'minDistance': 7, 'blockSize': 7
        })
        
        corners = cv2.goodFeaturesToTrack(gray, mask=self.roi_mask, **params)
        
        # Fallback to ORB if too few features
        min_pts = self.config.get('min_points', 30)
        if corners is None or len(corners) < min_pts:
            print("Falling back to ORB feature detection")
            orb = cv2.ORB_create(nfeatures=500)
            keypoints, _ = orb.detectAndCompute(gray, mask=self.roi_mask)
            if keypoints:
                corners = np.array([kp.pt for kp in keypoints], dtype=np.float32)
                corners = corners.reshape(-1, 1, 2)
        
        # If still insufficient, relax Shi-Tomasi parameters
        if corners is None or len(corners) < min_pts:
            params['qualityLevel'] = 0.001
            params['minDistance'] = 5
            corners = cv2.goodFeaturesToTrack(gray, mask=self.roi_mask, **params)
        
        return corners if corners is not None else np.array([]).reshape(-1, 1, 2)

    def filter_velocities_robust(self, arr, method='hybrid'):
        """Robust outlier filtering with hybrid MAD and IQR."""
        if arr is None or len(arr) < 3:
            return arr
        
        arr = np.array(arr)
        
        if method == 'hybrid':
            # MAD for non-Gaussian outliers
            median = np.median(arr)
            mad = np.median(np.abs(arr - median))
            mad_threshold = 3.5
            mad_mask = np.abs(arr - median) < mad_threshold * mad
            
            # IQR for additional robustness
            q25, q75 = np.percentile(arr, [25, 75])
            iqr = q75 - q25
            iqr_threshold = self.config.get('outlier_threshold', 1.5)
            iqr_mask = (arr >= q25 - iqr_threshold * iqr) & (arr <= q75 + iqr_threshold * iqr)
            
            # Combine masks
            combined_mask = mad_mask & iqr_mask
            return arr[combined_mask]
        
        return arr

    def pixel_to_mps(self, pix_val, fps):
        """Convert pixel displacement to meters per second."""
        mps = pix_val * fps * self.scale
        min_v = self.config.get('min_velocity_threshold', 0.005)
        max_v = self.config.get('max_velocity_threshold', 5.0)
        
        if np.isscalar(mps):
            return np.clip(mps, min_v, max_v) if mps > min_v else 0
        else:
            result = np.clip(mps, min_v, max_v)
            result[mps <= min_v] = 0
            return result

    def calculate_uncertainty(self, velocities, method='klt', num_points=0):
        """Enhanced uncertainty calculation with quality flags."""
        uncertainties = {}
        
        if velocities is None or len(velocities) == 0:
            return {
                'std_dev': 0.0, 'std_error': 0.0, 'confidence_95': 0.0,
                'coefficient_variation': 0.0, 'spatial_uncertainty': 0.0,
                'scale_uncertainty': 0.0, 'total_uncertainty': 0.0,
                'quality_flag': 'no_data'
            }
        
        velocities = np.array(velocities)
        valid_velocities = velocities[velocities > 0.001]
        
        if len(valid_velocities) == 0:
            return {
                'std_dev': 0.0, 'std_error': 0.0, 'confidence_95': 0.0,
                'coefficient_variation': 0.0, 'spatial_uncertainty': 0.0,
                'scale_uncertainty': 0.0, 'total_uncertainty': 0.0,
                'quality_flag': 'insufficient_signal'
            }
        
        uncertainties['std_dev'] = np.std(valid_velocities)
        uncertainties['std_error'] = uncertainties['std_dev'] / np.sqrt(len(valid_velocities))
        uncertainties['confidence_95'] = 1.96 * uncertainties['std_error'] if len(valid_velocities) > 1 else 0.0
        mean_val = np.mean(valid_velocities)
        uncertainties['coefficient_variation'] = uncertainties['std_dev'] / mean_val if mean_val > 0.001 else 999.0
        
        if method == 'klt':
            base_uncertainty = 0.008
            uncertainties['spatial_uncertainty'] = base_uncertainty / np.sqrt(max(num_points, 1))
        else:
            uncertainties['spatial_uncertainty'] = 0.003
        
        mean_pix_vel = mean_val / (self.scale * self.config.get('fps', 30.0))
        uncertainties['scale_uncertainty'] = mean_pix_vel * self.scale_uncertainty * self.config.get('fps', 30.0)
        
        uncertainties['total_uncertainty'] = np.sqrt(
            uncertainties['std_error']**2 +
            uncertainties['spatial_uncertainty']**2 +
            uncertainties['scale_uncertainty']**2
        )
        
        cv = uncertainties['coefficient_variation']
        if cv > 0.8 or (method == 'klt' and num_points < 10):
            uncertainties['quality_flag'] = 'poor'
        elif cv > 0.5 or (method == 'klt' and num_points < 20):
            uncertainties['quality_flag'] = 'fair'
        elif cv > 0.25:
            uncertainties['quality_flag'] = 'good'
        else:
            uncertainties['quality_flag'] = 'excellent'
        
        return uncertainties

    def combine_velocities(self, klt_mps, farne_mps, klt_uncertainty, farne_uncertainty):
        """Combine KLT and Farneback velocities based on quality."""
        klt_cv = klt_uncertainty['coefficient_variation']
        farne_cv = farne_uncertainty['coefficient_variation']
        
        # Weight inversely proportional to CV (lower CV = higher weight)
        klt_weight = 1 / (klt_cv + 0.01)  # Avoid division by zero
        farne_weight = 1 / (farne_cv + 0.01)
        total_weight = klt_weight + farne_weight
        
        if total_weight > 0:
            combined_mps = (klt_mps * klt_weight + farne_mps * farne_weight) / total_weight
            combined_uncertainty = np.sqrt(
                (klt_uncertainty['total_uncertainty'] * klt_weight) ** 2 +
                (farne_uncertainty['total_uncertainty'] * farne_weight) ** 2
            ) / total_weight
        else:
            combined_mps = farne_mps  # Fallback to Farneback
            combined_uncertainty = farne_uncertainty['total_uncertainty']
        
        return combined_mps, combined_uncertainty

    def visualize_flow(self, frame, mag, ang):
        """Enhanced flow visualization."""
        hsv = np.zeros_like(frame)
        hsv[..., 1] = 255
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return cv2.addWeighted(frame, 0.7, flow_vis, 0.3, 0)

    def compute_farneback_flow(self, old_gray, gray):
        """Enhanced Farneback with temporal smoothing."""
        fb_params = self.config.get('farneback_params', {})
        flow = cv2.calcOpticalFlowFarneback(old_gray, gray, None, **fb_params)
        
        alpha = self.config.get('temporal_smoothing', 0.3)
        if self.prev_flow is not None:
            flow = alpha * flow + (1 - alpha) * self.prev_flow
        self.prev_flow = flow.copy()
        
        return flow

    def validate_results(self, klt_mps, farne_mps, klt_uncertainty, farne_uncertainty):
        """Validate consistency between KLT and Farneback velocities."""
        if klt_mps == 0 or farne_mps == 0:
            return 'unreliable', 'Zero velocity detected'
        
        relative_diff = abs(klt_mps - farne_mps) / max(klt_mps, farne_mps)
        if relative_diff > 0.5:
            return 'unreliable', f'High discrepancy between KLT ({klt_mps:.4f}) and Farneback ({farne_mps:.4f})'
        
        if klt_uncertainty['quality_flag'] == 'poor' or farne_uncertainty['quality_flag'] == 'poor':
            return 'unreliable', 'Poor quality in one or both methods'
        
        return 'reliable', 'Consistent results'

    def process_video(self, video_path, roi_coords=None, perspective_pts=None, dst_size=(800, 400), reference_points=None, reference_length_m=None):
        """Process video with optimized algorithms."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.config['fps'] = fps
        
        # Estimate scale factor if reference provided
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Cannot read video")
        self.scale = self.estimate_scale_factor(frame, reference_points, reference_length_m)
        
        frame, M = self.perspective_warp(frame, perspective_pts, dst_size)
        self.create_roi_mask(frame.shape, roi_coords)
        old_gray = self.preprocess(frame)
        points = self.detect_features_adaptive(old_gray)
        
        results = []
        frame_num = 0
        color = np.random.randint(0, 255, (500, 3))
        
        window_size = self.config.get('uncertainty_window', 30)
        klt_velocity_window = deque(maxlen=window_size)
        farneback_velocity_window = deque(maxlen=window_size)
        combined_velocity_window = deque(maxlen=window_size)
        
        consecutive_poor_frames = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.warpPerspective(frame, M, dst_size) if perspective_pts is not None else frame
            gray = self.preprocess(frame)
            vis_frame = frame.copy()
            if roi_coords is not None:
                cv2.polylines(vis_frame, [np.array(roi_coords, dtype=np.int32)], True, (0, 255, 255), 2)
            
            klt_pix_vel = 0
            farne_pix_vel = 0
            npts = 0
            klt_raw_velocities = []
            
            # KLT Tracking
            if points is not None and len(points) > 0:
                lk_params = self.config.get('lk_params', {
                    'winSize': (25, 25), 'maxLevel': 4,
                    'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
                })
                
                new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray, points, None, **lk_params)
                
                if new_points is not None and status is not None:
                    error_threshold = self.config.get('klt_error_threshold', 20)
                    good_mask = (status.ravel() == 1) & (error.ravel() < error_threshold)
                    
                    good_new = new_points[good_mask]
                    good_old = points[good_mask]
                    npts = len(good_new)
                    
                    if npts > 0:
                        displacements = good_new - good_old
                        dists = np.linalg.norm(displacements, axis=2).flatten()
                        klt_raw_velocities = dists.copy()
                        dists_filtered = self.filter_velocities_robust(dists, method='hybrid')
                        
                        if len(dists_filtered) > 0:
                            klt_pix_vel = np.median(dists_filtered)
                        
                        for i, (new, old) in enumerate(zip(good_new, good_old)):
                            a, b = new.ravel().astype(int)
                            c, d = old.ravel().astype(int)
                            cv2.line(vis_frame, (a, b), (c, d), color[i % len(color)].tolist(), 2)
                            cv2.circle(vis_frame, (a, b), 3, color[i % len(color)].tolist(), -1)
                    
                    points = good_new.reshape(-1, 1, 2) if npts > 0 else None
                
                min_pts = self.config.get('min_points', 30)
                if npts < min_pts or consecutive_poor_frames > 5:
                    points = self.detect_features_adaptive(gray)
                    consecutive_poor_frames = 0
            else:
                points = self.detect_features_adaptive(gray)
            
            # Farneback Flow
            fb_flow = self.compute_farneback_flow(old_gray, gray)
            mag, ang = cv2.cartToPolar(fb_flow[..., 0], fb_flow[..., 1])
            
            if self.config.get('directional_filter', True):
                flow_direction = np.arctan2(fb_flow[..., 1], fb_flow[..., 0])
                direction_mask = np.abs(flow_direction) < np.pi/4
                mag_filtered = mag * direction_mask
            else:
                mag_filtered = mag
            
            mag_roi = mag_filtered[self.roi_mask > 0]
            valid = mag_roi[mag_roi > self.config.get('farneback_min_mag', 0.05)]
            valid_filtered = self.filter_velocities_robust(valid.flatten(), method='hybrid')
            
            if len(valid_filtered) > 0:
                farne_pix_vel = np.median(valid_filtered)
            
            # Convert to m/s
            klt_mps = self.pixel_to_mps(klt_pix_vel, fps)
            farne_mps = self.pixel_to_mps(farne_pix_vel, fps)
            
            # Calculate uncertainties
            klt_velocities_ms = self.pixel_to_mps(klt_raw_velocities, fps)
            klt_uncertainty = self.calculate_uncertainty(klt_velocities_ms, method='klt', num_points=npts)
            farneback_velocities_ms = self.pixel_to_mps(valid_filtered, fps)
            farneback_uncertainty = self.calculate_uncertainty(farneback_velocities_ms, method='farneback')
            
            # Combine velocities
            combined_mps, combined_uncertainty = self.combine_velocities(
                klt_mps, farne_mps, klt_uncertainty, farneback_uncertainty
            )
            
            # Update windows
            klt_velocity_window.append(klt_mps)
            farneback_velocity_window.append(farne_mps)
            combined_velocity_window.append(combined_mps)
            
            klt_window_uncertainty = self.calculate_uncertainty(list(klt_velocity_window), method='klt', num_points=npts)
            farneback_window_uncertainty = self.calculate_uncertainty(list(farneback_velocity_window), method='farneback')
            combined_window_uncertainty = self.calculate_uncertainty(list(combined_velocity_window), method='combined')
            
            # Validate results
            reliability, reliability_msg = self.validate_results(klt_mps, farne_mps, klt_uncertainty, farneback_uncertainty)
            if reliability == 'unreliable':
                consecutive_poor_frames += 1
            else:
                consecutive_poor_frames = 0
            
            # Visualization
            vis_frame = self.visualize_flow(vis_frame, mag_filtered, ang)
            cv2.putText(vis_frame, f"Combined: {combined_mps:.4f} +/- {combined_window_uncertainty['total_uncertainty']:.4f} m/s",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(vis_frame, f"KLT: {klt_mps:.4f} +/- {klt_window_uncertainty['total_uncertainty']:.4f} m/s",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 255), 2)
            cv2.putText(vis_frame, f"Farneback: {farne_mps:.4f} +/- {farneback_window_uncertainty['total_uncertainty']:.4f} m/s",
                        (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 255), 2)
            cv2.putText(vis_frame, f"Points: {npts} | Quality: {combined_window_uncertainty['quality_flag']}",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(vis_frame, f"Frame: {frame_num} | {reliability_msg}",
                        (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.imshow("River Velocity Analysis", vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            results.append({
                'frame': frame_num,
                'velocity_klt_m_per_s': klt_mps,
                'velocity_farneback_m_per_s': farne_mps,
                'velocity_combined_m_per_s': combined_mps,
                'num_klt_points': npts,
                'klt_uncertainty': klt_uncertainty['total_uncertainty'],
                'klt_cv': klt_uncertainty['coefficient_variation'],
                'klt_quality': klt_uncertainty['quality_flag'],
                'farneback_uncertainty': farneback_uncertainty['total_uncertainty'],
                'farneback_cv': farneback_uncertainty['coefficient_variation'],
                'farneback_quality': farneback_uncertainty['quality_flag'],
                'combined_uncertainty': combined_uncertainty,
                'combined_cv': combined_window_uncertainty['coefficient_variation'],
                'combined_quality': combined_window_uncertainty['quality_flag'],
                'reliability': reliability,
                'reliability_msg': reliability_msg
            })
            
            frame_num += 1
            old_gray = gray.copy()
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Post-processing
        df = pd.DataFrame(results)
        
        # Dynamic smoothing based on CV
        mean_cv = df['combined_cv'].mean()
        smooth_window = min(21, len(df) // 10)
        if mean_cv > 0.5:
            smooth_window = min(smooth_window + 4, len(df))  # Increase window for noisy data
        if smooth_window % 2 == 0:
            smooth_window += 1
        smooth_window = max(5, smooth_window)
        
        if len(df) >= smooth_window:
            df['velocity_klt_smoothed'] = savgol_filter(df['velocity_klt_m_per_s'],
                                                       window_length=smooth_window, polyorder=2)
            df['velocity_farneback_smoothed'] = savgol_filter(df['velocity_farneback_m_per_s'],
                                                             window_length=smooth_window, polyorder=2)
            df['velocity_combined_smoothed'] = savgol_filter(df['velocity_combined_m_per_s'],
                                                            window_length=smooth_window, polyorder=2)
        else:
            df['velocity_klt_smoothed'] = medfilt(df['velocity_klt_m_per_s'], kernel_size=5)
            df['velocity_farneback_smoothed'] = medfilt(df['velocity_farneback_m_per_s'], kernel_size=5)
            df['velocity_combined_smoothed'] = medfilt(df['velocity_combined_m_per_s'], kernel_size=5)
        
        return df

# Optimized Configuration
config = {
    'scale': 0.000708,  # m/px, provide accurate value or use reference points
    'scale_uncertainty': 0.00005,
    'min_points': 30,
    'outlier_threshold': 1.5,
    'min_velocity_threshold': 0.005,
    'max_velocity_threshold': 5.0,
    'clahe_grid': (8, 8),
    'bilateral_d': 9,
    'enable_sharpening': True,
    'uncertainty_window': 30,
    'temporal_smoothing': 0.3,
    'directional_filter': True,
    'klt_error_threshold': 20,
    'farneback_params': {
        'pyr_scale': 0.5,
        'levels': 5,
        'winsize': 31,
        'iterations': 5,
        'poly_n': 7,
        'poly_sigma': 1.5,
        'flags': 0
    },
    'farneback_min_mag': 0.05,
    'klt_params': {
        'maxCorners': 500,
        'qualityLevel': 0.005,
        'minDistance': 7,
        'blockSize': 7
    },
    'lk_params': {
        'winSize': (25, 25),
        'maxLevel': 4,
        'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    }
}

# Run Estimator
video_path = "data/samples/sample_river_1.mp4"
roi_coords = [(375, 241), (541, 236), (623, 392), (300, 392)]
perspective_pts = None
# Example reference points and length (customize based on your scene)
reference_points = [(100, 100), (200, 100)]  # Pixel coordinates of a known object
reference_length_m = 1.0  # Actual length in meters
output_csv = "velocity_output_no_correction.csv"

print("="*70)
print("PROCESSING VIDEO WITH OPTIMIZED ALGORITHMS (NO CORRECTION FACTOR)")
print("="*70)

estimator = RiverVelocityEstimator(config)
df = estimator.process_video(
    video_path,
    roi_coords=roi_coords,
    perspective_pts=perspective_pts,
    reference_points=reference_points,
    reference_length_m=reference_length_m
)

# Save results
df.to_csv(output_csv, index=False)

# Enhanced visualization
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Plot 1: Velocities with uncertainty
ax1 = axes[0]
ax1.plot(df["frame"], df["velocity_combined_smoothed"], label="Combined", color="purple", linewidth=2, alpha=0.8)
ax1.fill_between(df["frame"],
                 df["velocity_combined_smoothed"] - df["combined_uncertainty"],
                 df["velocity_combined_smoothed"] + df["combined_uncertainty"],
                 alpha=0.2, color="purple")
ax1.plot(df["frame"], df["velocity_klt_smoothed"], label="KLT", color="green", linewidth=1.5, alpha=0.6)
ax1.plot(df["frame"], df["velocity_farneback_smoothed"], label="Farneback", color="blue", linewidth=1.5, alpha=0.6)
ax1.set_xlabel("Frame", fontsize=11)
ax1.set_ylabel("Velocity (m/s)", fontsize=11)
ax1.set_title("River Surface Velocity with Uncertainty Bands", fontsize=13, fontweight='bold')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)

# Plot 2: Coefficient of Variation
ax2 = axes[1]
ax2.plot(df["frame"], df["combined_cv"].clip(0, 2) * 100, label="Combined CV", color="purple", alpha=0.7)
ax2.plot(df["frame"], df["klt_cv"].clip(0, 2) * 100, label="KLT CV", color="green", alpha=0.5)
ax2.plot(df["frame"], df["farneback_cv"].clip(0, 2) * 100, label="Farneback CV", color="blue", alpha=0.5)
ax2.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='Fair threshold (50%)')
ax2.set_xlabel("Frame", fontsize=11)
ax2.set_ylabel("Coefficient of Variation (%)", fontsize=11)
ax2.set_title("Measurement Uncertainty Over Time", fontsize=13, fontweight='bold')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 200])

# Plot 3: Tracking points and reliability
ax3 = axes[2]
ax3.plot(df["frame"], df["num_klt_points"], label="KLT Points", color="purple", linewidth=1.5)
ax3.axhline(y=config['min_points'], color='red', linestyle='--', alpha=0.5, label=f'Min threshold ({config["min_points"]})')
ax3.set_xlabel("Frame", fontsize=11)
ax3.set_ylabel("Number of Tracking Points", fontsize=11)
ax3.set_title("Feature Tracking Quality", fontsize=13, fontweight='bold')
ax3.legend(loc='best')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('velocity_analysis_no_correction.png', dpi=300, bbox_inches='tight')
plt.show()

# Summary Report
print("\n" + "="*70)
print("VELOCITY SUMMARY WITH UNCERTAINTY")
print("="*70)

for method, col in [
    ("Combined", "velocity_combined_smoothed"),
    ("KLT", "velocity_klt_smoothed"),
    ("Farneback", "velocity_farneback_smoothed")
]:
    mean_vel = df[col].mean()
    std_vel = df[col].std()
    median_vel = df[col].median()
    uncertainty_col = f"{method.lower()}_uncertainty"
    mean_uncertainty = df[uncertainty_col].mean()
    cv = df[f"{method.lower()}_cv"].mean()
    
    quality_col = f"{method.lower()}_quality"
    quality_counts = df[quality_col].value_counts()
    quality_str = ", ".join([f"{k}: {v}" for k, v in quality_counts.items()])
    
    print(f"\n{method} Method:")
    print(f" Mean Velocity: {mean_vel:.4f} ± {mean_uncertainty:.4f} m/s")
    print(f" Median Velocity: {median_vel:.4f} m/s")
    print(f" Std Deviation: {std_vel:.4f} m/s")
    print(f" Mean CV: {cv:.2%}" if cv < 10 else f" Mean CV: {cv:.2f} (high)")
    print(f" 95% Confidence: [{mean_vel - 1.96*mean_uncertainty:.4f}, {mean_vel + 1.96*mean_uncertainty:.4f}] m/s")
    print(f" Quality: {quality_str}")
    
    if method == "KLT":
        print(f" Avg Points: {df['num_klt_points'].mean():.0f}")

# Reliability Assessment
good_frames_pct = len(df[df['combined_quality'].isin(['good', 'excellent'])]) / len(df) * 100
reliable_frames_pct = len(df[df['reliability'] == 'reliable']) / len(df) * 100
print(f"\n{'='*70}")
print("FINAL RECOMMENDATION")
print("="*70)
combined_mean = df['velocity_combined_smoothed'].mean()
combined_median = df['velocity_combined_smoothed'].median()
combined_cv = df['combined_cv'].mean()
print(f"\n✓ BEST ESTIMATE (Combined method):")
print(f" Mean: {combined_mean:.4f} ± {df['combined_uncertainty'].mean():.4f} m/s")
print(f" Median: {combined_median:.4f} m/s")
print(f"\n Measurement reliability: {good_frames_pct:.1f}% good/excellent frames")
print(f" Consistent KLT/Farneback: {reliable_frames_pct:.1f}% reliable frames")
if combined_cv < 0.3:
    print(f" Confidence level: HIGH (CV < 30%)")
elif combined_cv < 0.5:
    print(f" Confidence level: MODERATE (CV < 50%)")
else:
    print(f" Confidence level: LOW (CV > 50%)")

# Optimization Suggestions
print(f"\n{'='*70}")
print("OPTIMIZATION SUGGESTIONS")
print("="*70)
suggestions = []
if df['num_klt_points'].mean() < 50:
    suggestions.append("• Improve KLT tracking: Increase ROI size or enhance lighting")
if combined_cv > 0.5:
    suggestions.append("• High variability: Check for camera vibration or unstable flow")
    suggestions.append("• Consider increasing temporal_smoothing (currently 0.3)")
if good_frames_pct < 70:
    suggestions.append("• Low quality frame percentage: Review video quality and ROI placement")
    suggestions.append("• Ensure ROI covers actively flowing water with visible features")
if reliable_frames_pct < 70:
    suggestions.append("• Low consistency between KLT and Farneback: Check feature detection")
if df['velocity_combined_smoothed'].std() > 0.05:
    suggestions.append("• High velocity variability detected")
    suggestions.append("• Consider longer averaging window or multiple ROIs")
if reference_points is None or reference_length_m is None:
    suggestions.append("• Provide reference_points and reference_length_m for accurate scale")
if len(suggestions) == 0:
    print("\n✓ Measurements are optimal! No improvements needed.")
else:
    print("\nRecommended improvements:")
    for s in suggestions:
        print(f" {s}")

# Processing Statistics
print(f"\n{'='*70}")
print("PROCESSING STATISTICS")
print("="*70)
print(f" Total frames processed: {len(df)}")
print(f" Video duration: {len(df) / config['fps']:.1f} seconds")
print(f" Processing quality:")
print(f" - Excellent: {len(df[df['combined_quality'] == 'excellent'])}")
print(f" - Good: {len(df[df['combined_quality'] == 'good'])}")
print(f" - Fair: {len(df[df['combined_quality'] == 'fair'])}")
print(f" - Poor: {len(df[df['combined_quality'] == 'poor'])}")
print(f" Scale factor used: {config['scale']:.6f} m/px")
print("\n" + "="*70)
print(f"Results saved to: {output_csv}")
print(f"Visualization saved to: velocity_analysis_no_correction.png")
print("="*70)