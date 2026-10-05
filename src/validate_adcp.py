import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

print("="*70)
print("OPTICAL FLOW vs ADCP GROUND TRUTH COMPARISON")
print("="*70)

# Load optical flow results
import argparse
_p = argparse.ArgumentParser(description="Compare optical-flow velocities with ADCP ground truth")
_p.add_argument("--flow-csv", default="results/velocity_output_final.csv")
_p.add_argument("--adcp-dir", default="data/opara/FreibergerMulde/ADCP")
_args = _p.parse_args()
optical_flow_file = _args.flow_csv
if not os.path.exists(optical_flow_file):
    print(f"Error: {optical_flow_file} not found")
    exit()

df_optical = pd.read_csv(optical_flow_file)

print(f"\n✓ Loaded optical flow results: {len(df_optical)} frames")
print(f"  KLT mean velocity: {df_optical['velocity_klt_m_per_s'].mean():.4f} m/s")
print(f"  Farneback mean velocity: {df_optical['velocity_farneback_m_per_s'].mean():.4f} m/s")

# Parse ADCP data
adcp_dir = _args.adcp_dir
adcp_files = [
    "Profil1_v_tot.txt",
    "profil2_v_tot.txt", 
    "profil3_v_tot.txt"
]

print("\n" + "="*70)
print("PARSING ADCP DATA")
print("="*70)

adcp_results = {}

for filename in adcp_files:
    filepath = os.path.join(adcp_dir, filename)
    if not os.path.exists(filepath):
        continue
    
    print(f"\nProcessing: {filename}")
    
    try:
        # Read file with latin-1 encoding
        with open(filepath, 'r', encoding='latin-1') as f:
            lines = f.readlines()
        
        # Parse the semicolon-separated format
        # Line 1: Header (title)
        # Line 2: Distance from shore (meters)
        # Line 3: Fixed depth value (50.00)
        # Line 4: Reference depth/distance values
        # Line 5+: Time stamps
        # Following lines: Depth rows with velocity data
        
        velocities = []
        
        # Start from line 6 (after header lines)
        for line in lines[5:]:
            parts = line.strip().split(';')
            if len(parts) > 2:
                # First column is depth, rest are velocity values
                depth = parts[0]
                for vel_str in parts[2:]:  # Skip first two columns
                    vel_str = vel_str.replace(',', '.')  # Convert European decimal
                    try:
                        vel = float(vel_str)
                        if 0.01 <= vel <= 3.0:  # Reasonable velocity range
                            velocities.append(vel)
                    except:
                        continue
        
        if len(velocities) > 0:
            velocities = np.array(velocities)
            adcp_results[filename] = {
                'velocities': velocities,
                'mean': np.mean(velocities),
                'median': np.median(velocities),
                'std': np.std(velocities),
                'min': np.min(velocities),
                'max': np.max(velocities),
                'count': len(velocities)
            }
            
            print(f"  ✓ Parsed {len(velocities)} velocity measurements")
            print(f"    Mean: {np.mean(velocities):.4f} m/s")
            print(f"    Median: {np.median(velocities):.4f} m/s")
            print(f"    Std: {np.std(velocities):.4f} m/s")
            print(f"    Range: {np.min(velocities):.4f} - {np.max(velocities):.4f} m/s")
        
    except Exception as e:
        print(f"  ✗ Error parsing: {e}")

# Calculate overall ADCP statistics
if len(adcp_results) > 0:
    all_adcp_velocities = np.concatenate([r['velocities'] for r in adcp_results.values()])
    adcp_mean = np.mean(all_adcp_velocities)
    adcp_median = np.median(all_adcp_velocities)
    adcp_std = np.std(all_adcp_velocities)
    
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    # Optical flow statistics
    klt_mean = df_optical['velocity_klt_m_per_s'].mean()
    klt_median = df_optical['velocity_klt_m_per_s'].median()
    klt_std = df_optical['velocity_klt_m_per_s'].std()
    
    farneback_mean = df_optical['velocity_farneback_m_per_s'].mean()
    farneback_median = df_optical['velocity_farneback_m_per_s'].median()
    farneback_std = df_optical['velocity_farneback_m_per_s'].std()
    
    print(f"\nOptical Flow (Surface Velocities from Video):")
    print(f"  KLT Method:")
    print(f"    Mean:   {klt_mean:.4f} m/s (±{klt_std:.4f})")
    print(f"    Median: {klt_median:.4f} m/s")
    
    print(f"\n  Farneback Method:")
    print(f"    Mean:   {farneback_mean:.4f} m/s (±{farneback_std:.4f})")
    print(f"    Median: {farneback_median:.4f} m/s")
    
    print(f"\nADCP Ground Truth (3D Flow Profile):")
    print(f"  Combined from {len(adcp_results)} profiles:")
    print(f"    Mean:   {adcp_mean:.4f} m/s (±{adcp_std:.4f})")
    print(f"    Median: {adcp_median:.4f} m/s")
    print(f"    Total measurements: {len(all_adcp_velocities)}")
    
    # Calculate errors
    klt_error = abs(klt_mean - adcp_mean) / adcp_mean * 100
    farneback_error = abs(farneback_mean - adcp_mean) / adcp_mean * 100
    
    print(f"\n" + "─"*70)
    print(f"Relative Error vs ADCP Mean ({adcp_mean:.4f} m/s):")
    print(f"─"*70)
    print(f"  KLT:       {klt_error:>6.2f}% error ({klt_mean - adcp_mean:+.4f} m/s)")
    print(f"  Farneback: {farneback_error:>6.2f}% error ({farneback_mean - adcp_mean:+.4f} m/s)")
    
    # Visualization
    print("\n" + "="*70)
    print("CREATING COMPARISON PLOTS")
    print("="*70)
    
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: Bar chart comparison
    ax1 = plt.subplot(2, 2, 1)
    methods = ['KLT\n(Optical Flow)', 'Farneback\n(Optical Flow)', 'ADCP\n(Ground Truth)']
    means = [klt_mean, farneback_mean, adcp_mean]
    stds = [klt_std, farneback_std, adcp_std]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    bars = ax1.bar(methods, means, color=colors, alpha=0.7, width=0.6)
    ax1.errorbar(methods, means, yerr=stds, fmt='none', color='black', 
                 capsize=8, linewidth=2, label='Std Dev')
    
    ax1.set_ylabel('Mean Velocity (m/s)', fontsize=12, fontweight='bold')
    ax1.set_title('Surface Velocity Comparison', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend()
    
    # Add value labels on bars
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.3f} m/s', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Velocity distributions
    ax2 = plt.subplot(2, 2, 2)
    ax2.hist(df_optical['velocity_klt_m_per_s'], bins=30, alpha=0.5, 
             label=f'KLT (μ={klt_mean:.3f})', color='blue', density=True)
    ax2.hist(df_optical['velocity_farneback_m_per_s'], bins=30, alpha=0.5,
             label=f'Farneback (μ={farneback_mean:.3f})', color='red', density=True)
    ax2.hist(all_adcp_velocities, bins=30, alpha=0.5,
             label=f'ADCP (μ={adcp_mean:.3f})', color='green', density=True)
    
    ax2.set_xlabel('Velocity (m/s)', fontsize=11)
    ax2.set_ylabel('Probability Density', fontsize=11)
    ax2.set_title('Velocity Distribution Comparison', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Box plots
    ax3 = plt.subplot(2, 2, 3)
    data_to_plot = [
        df_optical['velocity_klt_m_per_s'].values,
        df_optical['velocity_farneback_m_per_s'].values,
        all_adcp_velocities
    ]
    box = ax3.boxplot(data_to_plot, labels=methods, patch_artist=True)
    
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax3.set_ylabel('Velocity (m/s)', fontsize=11, fontweight='bold')
    ax3.set_title('Velocity Statistics (Box Plot)', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Error comparison
    ax4 = plt.subplot(2, 2, 4)
    errors = [klt_error, farneback_error]
    error_methods = ['KLT', 'Farneback']
    error_colors = ['#3498db', '#e74c3c']
    
    bars = ax4.bar(error_methods, errors, color=error_colors, alpha=0.7)
    ax4.axhline(y=20, color='orange', linestyle='--', linewidth=2, 
                label='±20% threshold')
    ax4.axhline(y=10, color='green', linestyle='--', linewidth=2,
                label='±10% threshold')
    
    ax4.set_ylabel('Relative Error (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Error vs ADCP Ground Truth', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, error in zip(bars, errors):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{error:.1f}%', ha='center', va='bottom', 
                fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('velocity_comparison_plot.png', dpi=150, bbox_inches='tight')
    print("✓ Saved comparison plot to 'velocity_comparison_plot.png'")
    plt.show()
    
    # Save detailed comparison CSV
    comparison_df = pd.DataFrame({
        'Method': ['KLT', 'Farneback', 'ADCP'],
        'Mean_Velocity_ms': [klt_mean, farneback_mean, adcp_mean],
        'Median_Velocity_ms': [klt_median, farneback_median, adcp_median],
        'Std_Dev_ms': [klt_std, farneback_std, adcp_std],
        'Error_vs_ADCP_%': [klt_error, farneback_error, 0.0]
    })
    
    comparison_df.to_csv('combined_velocity_comparison.csv', index=False)
    print("✓ Saved detailed comparison to 'combined_velocity_comparison.csv'")
    
else:
    print("\n✗ No ADCP data could be parsed")

print("\n" + "="*70)
print("Analysis complete!")
print("="*70)