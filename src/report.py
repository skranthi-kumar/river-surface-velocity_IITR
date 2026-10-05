"""
Complete Surface Velocity Analysis Pipeline
============================================
Integrates: Calibration → Velocity Estimation → Validation → Reporting
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

class SurfaceVelocityAnalyzer:
    def __init__(self, video_path, scale, fps=None):
        self.video_path = video_path
        self.scale = scale  # m/px
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        self.fps = fps if fps else self.cap.get(cv2.CAP_PROP_FPS)
        self.dt = 1.0 / self.fps
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print("="*70)
        print("SURFACE VELOCITY ANALYZER INITIALIZED")
        print("="*70)
        print(f"Video: {os.path.basename(video_path)}")
        print(f"Resolution: {self.width}x{self.height}")
        print(f"FPS: {self.fps:.2f}")
        print(f"Total frames: {self.total_frames}")
        print(f"Scale: {scale:.6f} m/px (GSD: {scale*100:.2f} cm/pixel)")
        print("="*70)
        
    def analyze(self, method='both', show_preview=True, outlier_filter=True):
        """
        Main analysis method
        
        Parameters:
        -----------
        method : str
            'klt', 'farneback', or 'both'
        show_preview : bool
            Display real-time visualization
        outlier_filter : bool
            Apply outlier rejection
        """
        
        # Feature detection parameters
        feature_params = dict(
            maxCorners=300,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=7
        )
        
        # LK parameters
        lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        # Initialize
        ret, old_frame = self.cap.read()
        if not ret:
            raise ValueError("Cannot read first frame")
        
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        
        # CLAHE for better feature detection
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        old_gray = clahe.apply(old_gray)
        
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        
        results = []
        frame_num = 0
        MIN_POINTS = 50
        
        print("\nProcessing video...")
        print("Press 'q' to stop\n")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_gray = clahe.apply(frame_gray)
            
            result = {
                'frame': frame_num,
                'time_s': frame_num / self.fps
            }
            
            # === KLT Optical Flow ===
            if method in ['klt', 'both'] and p0 is not None:
                vel_klt, num_points, p0 = self._process_klt(
                    old_gray, frame_gray, p0, lk_params, 
                    feature_params, outlier_filter, MIN_POINTS
                )
                result['velocity_klt_m_per_s'] = vel_klt
                result['num_klt_points'] = num_points
            
            # === Farneback Optical Flow ===
            if method in ['farneback', 'both']:
                vel_farneback = self._process_farneback(
                    old_gray, frame_gray, outlier_filter
                )
                result['velocity_farneback_m_per_s'] = vel_farneback
            
            results.append(result)
            
            # Visualization
            if show_preview:
                self._show_preview(frame, result, frame_num)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nStopped by user")
                    break
            
            old_gray = frame_gray.copy()
            frame_num += 1
            
            if frame_num % 50 == 0:
                print(f"  Processed {frame_num}/{self.total_frames} frames "
                      f"({frame_num/self.total_frames*100:.1f}%)")
        
        self.cap.release()
        if show_preview:
            cv2.destroyAllWindows()
        
        print(f"\n✓ Completed: {frame_num} frames processed")
        
        return pd.DataFrame(results)
    
    def _process_klt(self, old_gray, frame_gray, p0, lk_params, 
                     feature_params, outlier_filter, MIN_POINTS):
        """Process KLT optical flow with outlier rejection"""
        
        if len(p0) == 0:
            return 0.0, 0, None
        
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params
        )
        
        if p1 is None or st is None:
            return 0.0, 0, None
        
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        
        if len(good_new) == 0:
            return 0.0, 0, None
        
        # Calculate displacements
        displacements = good_new - good_old
        dists = np.linalg.norm(displacements, axis=1)
        
        # Outlier rejection
        if outlier_filter and len(dists) > 5:
            q1 = np.percentile(dists, 25)
            q3 = np.percentile(dists, 75)
            iqr = q3 - q1
            lower = max(0, q1 - 1.5 * iqr)
            upper = q3 + 1.5 * iqr
            
            valid_mask = (dists >= lower) & (dists <= upper)
            dists = dists[valid_mask]
            good_new = good_new[valid_mask]
        
        if len(dists) < 5:
            return 0.0, 0, None
        
        # Use median (robust to outliers)
        vel_px = np.median(dists) / self.dt
        vel_m_per_s = vel_px * self.scale
        
        # Update points
        p0_new = good_new.reshape(-1, 1, 2)
        
        # Re-detect if too few
        if len(p0_new) < MIN_POINTS:
            p0_new = cv2.goodFeaturesToTrack(
                frame_gray, mask=None, **feature_params
            )
        
        return vel_m_per_s, len(good_new), p0_new
    
    def _process_farneback(self, old_gray, frame_gray, outlier_filter):
        """Process Farneback optical flow"""
        
        flow = cv2.calcOpticalFlowFarneback(
            old_gray, frame_gray, None,
            pyr_scale=0.5, levels=3, winsize=21,
            iterations=5, poly_n=7, poly_sigma=1.5, flags=0
        )
        
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Filter outliers
        if outlier_filter:
            mag_flat = mag.flatten()
            mag_filtered = mag_flat[
                (mag_flat > np.percentile(mag_flat, 10)) & 
                (mag_flat < np.percentile(mag_flat, 90))
            ]
            vel_px = np.median(mag_filtered) / self.dt
        else:
            vel_px = np.mean(mag) / self.dt
        
        return vel_px * self.scale
    
    def _show_preview(self, frame, result, frame_num):
        """Display preview window"""
        
        display = frame.copy()
        y_offset = 30
        
        cv2.putText(display, f"Frame: {frame_num}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 30
        
        if 'velocity_klt_m_per_s' in result:
            cv2.putText(display, f"KLT: {result['velocity_klt_m_per_s']:.3f} m/s",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            y_offset += 30
        
        if 'velocity_farneback_m_per_s' in result:
            cv2.putText(display, f"Farneback: {result['velocity_farneback_m_per_s']:.3f} m/s",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow("Velocity Analysis", display)
    
    def generate_report(self, df, output_dir='.', adcp_mean=None):
        """Generate analysis report with plots"""
        
        print("\n" + "="*70)
        print("GENERATING ANALYSIS REPORT")
        print("="*70)
        
        # Statistics
        stats = {}
        
        if 'velocity_klt_m_per_s' in df.columns:
            stats['KLT'] = {
                'mean': df['velocity_klt_m_per_s'].mean(),
                'median': df['velocity_klt_m_per_s'].median(),
                'std': df['velocity_klt_m_per_s'].std(),
                'min': df['velocity_klt_m_per_s'].min(),
                'max': df['velocity_klt_m_per_s'].max()
            }
        
        if 'velocity_farneback_m_per_s' in df.columns:
            stats['Farneback'] = {
                'mean': df['velocity_farneback_m_per_s'].mean(),
                'median': df['velocity_farneback_m_per_s'].median(),
                'std': df['velocity_farneback_m_per_s'].std(),
                'min': df['velocity_farneback_m_per_s'].min(),
                'max': df['velocity_farneback_m_per_s'].max()
            }
        
        # Print statistics
        print("\nVelocity Statistics:")
        print("─"*70)
        for method, values in stats.items():
            print(f"\n{method} Method:")
            print(f"  Mean:   {values['mean']:.4f} m/s")
            print(f"  Median: {values['median']:.4f} m/s")
            print(f"  Std:    {values['std']:.4f} m/s")
            print(f"  Range:  {values['min']:.4f} - {values['max']:.4f} m/s")
        
        # Error analysis
        if adcp_mean:
            print("\n" + "─"*70)
            print(f"Comparison with ADCP ({adcp_mean:.4f} m/s):")
            print("─"*70)
            for method, values in stats.items():
                error = abs(values['mean'] - adcp_mean) / adcp_mean * 100
                print(f"  {method}: {error:>6.2f}% error")
        
        # Create plots
        self._create_plots(df, stats, output_dir, adcp_mean)
        
        # Save CSV
        csv_path = os.path.join(output_dir, 'velocity_results_final.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Results saved to: {csv_path}")
        
        return stats
    
    def _create_plots(self, df, stats, output_dir, adcp_mean):
        """Create comprehensive plots"""
        
        fig = plt.figure(figsize=(16, 12))
        
        # Plot 1: Time series
        ax1 = plt.subplot(3, 2, 1)
        if 'velocity_klt_m_per_s' in df.columns:
            ax1.plot(df['frame'], df['velocity_klt_m_per_s'], 
                    label='KLT', alpha=0.7, color='blue')
        if 'velocity_farneback_m_per_s' in df.columns:
            ax1.plot(df['frame'], df['velocity_farneback_m_per_s'],
                    label='Farneback', alpha=0.7, color='red')
        if adcp_mean:
            ax1.axhline(adcp_mean, color='green', linestyle='--', 
                       label=f'ADCP: {adcp_mean:.3f} m/s')
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Velocity (m/s)')
        ax1.set_title('Velocity Time Series')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Histograms
        ax2 = plt.subplot(3, 2, 2)
        if 'velocity_klt_m_per_s' in df.columns:
            ax2.hist(df['velocity_klt_m_per_s'], bins=30, alpha=0.5,
                    label='KLT', color='blue', density=True)
        if 'velocity_farneback_m_per_s' in df.columns:
            ax2.hist(df['velocity_farneback_m_per_s'], bins=30, alpha=0.5,
                    label='Farneback', color='red', density=True)
        ax2.set_xlabel('Velocity (m/s)')
        ax2.set_ylabel('Probability Density')
        ax2.set_title('Velocity Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Cumulative distribution
        ax3 = plt.subplot(3, 2, 3)
        if 'velocity_klt_m_per_s' in df.columns:
            sorted_klt = np.sort(df['velocity_klt_m_per_s'])
            ax3.plot(sorted_klt, np.linspace(0, 1, len(sorted_klt)),
                    label='KLT', color='blue')
        if 'velocity_farneback_m_per_s' in df.columns:
            sorted_fb = np.sort(df['velocity_farneback_m_per_s'])
            ax3.plot(sorted_fb, np.linspace(0, 1, len(sorted_fb)),
                    label='Farneback', color='red')
        ax3.set_xlabel('Velocity (m/s)')
        ax3.set_ylabel('Cumulative Probability')
        ax3.set_title('Cumulative Distribution Function')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Box plot
        ax4 = plt.subplot(3, 2, 4)
        data = []
        labels = []
        if 'velocity_klt_m_per_s' in df.columns:
            data.append(df['velocity_klt_m_per_s'])
            labels.append('KLT')
        if 'velocity_farneback_m_per_s' in df.columns:
            data.append(df['velocity_farneback_m_per_s'])
            labels.append('Farneback')
        if data:
            ax4.boxplot(data, tick_labels=labels)
        if adcp_mean:
            ax4.axhline(adcp_mean, color='green', linestyle='--',
                       label=f'ADCP: {adcp_mean:.3f}')
            ax4.legend()
        ax4.set_ylabel('Velocity (m/s)')
        ax4.set_title('Statistical Summary')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Plot 5: Feature tracking quality
        if 'num_klt_points' in df.columns:
            ax5 = plt.subplot(3, 2, 5)
            ax5.plot(df['frame'], df['num_klt_points'], color='green')
            ax5.set_xlabel('Frame')
            ax5.set_ylabel('Features Tracked')
            ax5.set_title('KLT Tracking Quality')
            ax5.grid(True, alpha=0.3)
        
        # Plot 6: Error vs ADCP
        if adcp_mean:
            ax6 = plt.subplot(3, 2, 6)
            errors = []
            methods_list = []
            for method, values in stats.items():
                error = abs(values['mean'] - adcp_mean) / adcp_mean * 100
                errors.append(error)
                methods_list.append(method)
            
            bars = ax6.bar(methods_list, errors, alpha=0.7)
            ax6.axhline(20, color='orange', linestyle='--', label='±20%')
            ax6.axhline(10, color='green', linestyle='--', label='±10%')
            ax6.set_ylabel('Relative Error (%)')
            ax6.set_title('Error vs ADCP Ground Truth')
            ax6.legend()
            ax6.grid(True, alpha=0.3, axis='y')
            
            for bar, error in zip(bars, errors):
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height,
                        f'{error:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'velocity_analysis_complete.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plots saved to: {plot_path}")
        plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # Configuration
    VIDEO_PATH = "oparaFiles/FreibergerMulde/terrestrial/CIMG3606.MOV"
    SCALE = 0.046455  # m/px (from GCP calibration)
    ADCP_MEAN = 0.5884  # m/s (from ground truth)
    
    # Initialize analyzer
    analyzer = SurfaceVelocityAnalyzer(VIDEO_PATH, SCALE)
    
    # Run analysis
    df = analyzer.analyze(
        method='both',           # 'klt', 'farneback', or 'both'
        show_preview=True,       # Show real-time visualization
        outlier_filter=True      # Apply outlier rejection
    )
    
    # Generate report
    stats = analyzer.generate_report(
        df, 
        output_dir='.',
        adcp_mean=ADCP_MEAN
    )
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  ✓ velocity_results_final.csv")
    print("  ✓ velocity_analysis_complete.png")
    print("\n" + "="*70)