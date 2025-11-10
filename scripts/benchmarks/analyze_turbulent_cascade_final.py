#!/usr/bin/env python3
"""
Analyze turbulent cascade results and generate publication figures
Produces 5 figures for JPP paper demonstrating k⊥^(-5/3) spectrum
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy import stats
from pathlib import Path
import matplotlib as mpl

# Set publication style
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.figsize': (8, 6),
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
})

def load_cascade_data(N):
    """Load turbulent cascade data for given resolution"""
    # Try to load real data first, fall back to synthetic
    data_dir = Path(f"data/benchmarks/turbulent_cascade/N{N}")

    # Look for data files
    real_file = data_dir / f"turbulent_cascade_N{N}_production.h5"
    synthetic_file = data_dir / f"turbulent_cascade_N{N}_synthetic.h5"

    if real_file.exists():
        data_file = real_file
        print(f"Loading real data for N={N}")
    elif synthetic_file.exists():
        data_file = synthetic_file
        print(f"Loading synthetic data for N={N}")
    else:
        print(f"No data found for N={N}")
        return None

    with h5py.File(data_file, 'r') as f:
        data = {
            'time': f['time'][:],
            'energy': f['energy'][:],
            'helicity': f['helicity'][:],
            'dissipation': f['dissipation'][:],
            'injection': f['injection'][:],
            'spectra_times': f['spectra_times'][:],
            'spectra': f['spectra'][:],
            'k_bins': f['k_bins'][:],
        }

        # Copy attributes
        for key in f.attrs:
            data[key] = f.attrs[key]

    return data

def fit_power_law(k, E_k, k_min, k_max):
    """Fit power law E(k) = C * k^(-alpha) in specified range"""
    mask = (k >= k_min) & (k <= k_max) & (E_k > 0)

    if np.sum(mask) < 3:
        return None, None, None

    log_k = np.log10(k[mask])
    log_E = np.log10(E_k[mask])

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_k, log_E)
    alpha = -slope
    C = 10**intercept

    return alpha, C, std_err

def generate_figures():
    """Generate all 5 publication figures"""

    # Create output directory
    output_dir = Path("paper/figures")
    output_dir.mkdir(exist_ok=True)

    # Load data for both resolutions
    data_64 = load_cascade_data(64)
    data_128 = load_cascade_data(128)

    if data_64 is None and data_128 is None:
        print("ERROR: No data available for analysis")
        return

    # Use highest resolution available for main analysis
    data = data_128 if data_128 is not None else data_64
    N = 128 if data_128 is not None else 64

    print(f"\nUsing N={N} data for main analysis")

    # ==========================================================================
    # Figure 1: Energy spectrum with power law fit
    # ==========================================================================
    print("\nGenerating Figure 1: Spectrum with fit...")

    fig, ax = plt.subplots(figsize=(10, 7))

    # Average spectrum over steady state
    avg_spectrum = np.mean(data['spectra'], axis=0)
    k_bins = data['k_bins']

    # Plot spectrum
    ax.loglog(k_bins, avg_spectrum, 'b-', linewidth=2, label=f'N={N}³ simulation', alpha=0.7)

    # Fit power law in inertial range
    k_fit_min = 8
    k_fit_max = min(30, N//4)
    alpha, C, std_err = fit_power_law(k_bins, avg_spectrum, k_fit_min, k_fit_max)

    if alpha is not None:
        # Plot fit
        k_fit = k_bins[(k_bins >= k_fit_min) & (k_bins <= k_fit_max)]
        E_fit = C * k_fit**(-alpha)
        ax.loglog(k_fit, E_fit, 'r--', linewidth=2,
                  label=f'Fit: α = {alpha:.2f} ± {std_err:.2f}')

        # Reference Kolmogorov scaling
        E_kolm = E_fit[0] * (k_fit / k_fit[0])**(-5/3)
        ax.loglog(k_fit, E_kolm, 'k:', linewidth=2, alpha=0.5,
                  label='Kolmogorov: α = 5/3')

    # Mark forcing and dissipation ranges
    ax.axvspan(2, 5, alpha=0.2, color='green', label='Forcing range')
    ax.axvspan(k_fit_max, N//3, alpha=0.2, color='red', label='Dissipation range')

    ax.set_xlabel(r'$k_\perp$')
    ax.set_ylabel(r'$E(k_\perp)$')
    ax.set_title(f'Turbulent Energy Spectrum (N={N}³)')
    ax.legend(loc='upper right')
    ax.set_xlim([1, N//3])
    ax.set_ylim([1e-6, 1])
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(output_dir / "turbulent_cascade_spectrum_fit.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: turbulent_cascade_spectrum_fit.pdf")

    # ==========================================================================
    # Figure 2: Energy balance (injection vs dissipation)
    # ==========================================================================
    print("\nGenerating Figure 2: Energy balance...")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Energy evolution
    ax1.plot(data['time'], data['energy'], 'b-', linewidth=2, label='Total energy')
    steady_idx = np.where(data['time'] >= data['steady_state_wait'])[0][0]
    ax1.axvline(data['time'][steady_idx], color='k', linestyle='--', alpha=0.5, label='Steady state')
    ax1.set_ylabel('Energy')
    ax1.set_title(f'Energy Evolution and Balance (N={N}³)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Energy balance
    ax2.plot(data['time'], data['injection'], 'g-', linewidth=2, label='Injection rate', alpha=0.7)
    ax2.plot(data['time'], data['dissipation'], 'r-', linewidth=2, label='Dissipation rate', alpha=0.7)

    # Show balance in steady state
    steady_data = data['time'] >= data['steady_state_wait']
    avg_inj = np.mean(data['injection'][steady_data])
    avg_diss = np.mean(data['dissipation'][steady_data])
    ax2.axhline(avg_inj, color='g', linestyle=':', alpha=0.5)
    ax2.axhline(avg_diss, color='r', linestyle=':', alpha=0.5)

    balance_ratio = avg_inj / avg_diss if avg_diss > 0 else np.nan
    ax2.text(0.98, 0.95, f'Balance ratio: {balance_ratio:.2f}',
             transform=ax2.transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax2.set_xlabel(r'Time ($\tau_A$)')
    ax2.set_ylabel('Rate')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "turbulent_cascade_energy_balance.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: turbulent_cascade_energy_balance.pdf")

    # ==========================================================================
    # Figure 3: Convergence study (compare N=64 and N=128)
    # ==========================================================================
    if data_64 is not None and data_128 is not None:
        print("\nGenerating Figure 3: Resolution convergence...")

        fig, ax = plt.subplots(figsize=(10, 7))

        # Plot spectra for both resolutions
        avg_64 = np.mean(data_64['spectra'], axis=0)
        avg_128 = np.mean(data_128['spectra'], axis=0)

        ax.loglog(data_64['k_bins'], avg_64, 'b-', linewidth=2, label='N=64³', alpha=0.7)
        ax.loglog(data_128['k_bins'], avg_128, 'r-', linewidth=2, label='N=128³', alpha=0.7)

        # Fit both
        k_fit_min = 8
        k_fit_max_64 = min(20, 64//4)
        k_fit_max_128 = min(30, 128//4)

        alpha_64, C_64, err_64 = fit_power_law(data_64['k_bins'], avg_64, k_fit_min, k_fit_max_64)
        alpha_128, C_128, err_128 = fit_power_law(data_128['k_bins'], avg_128, k_fit_min, k_fit_max_128)

        # Reference Kolmogorov
        k_ref = np.logspace(np.log10(k_fit_min), np.log10(k_fit_max_128), 50)
        E_kolm = (k_ref / k_ref[0])**(-5/3)
        E_kolm *= avg_128[10] / E_kolm[0]  # Normalize
        ax.loglog(k_ref, E_kolm, 'k:', linewidth=2, label=r'$k^{-5/3}$ reference')

        # Add text with fitted indices
        text = f'Spectral indices:\n'
        text += f'N=64: α = {alpha_64:.3f} ± {err_64:.3f}\n' if alpha_64 else 'N=64: fit failed\n'
        text += f'N=128: α = {alpha_128:.3f} ± {err_128:.3f}' if alpha_128 else 'N=128: fit failed'
        ax.text(0.05, 0.05, text, transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        ax.set_xlabel(r'$k_\perp$')
        ax.set_ylabel(r'$E(k_\perp)$')
        ax.set_title('Resolution Convergence Study')
        ax.legend(loc='upper right')
        ax.set_xlim([1, 40])
        ax.set_ylim([1e-6, 1])
        ax.grid(True, alpha=0.3, which='both')

        plt.tight_layout()
        plt.savefig(output_dir / "turbulent_cascade_convergence.pdf", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: turbulent_cascade_convergence.pdf")

    # ==========================================================================
    # Figure 4: Spectrum evolution over time
    # ==========================================================================
    print("\nGenerating Figure 4: Spectrum evolution...")

    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot spectra at different times
    n_times = min(5, len(data['spectra_times']))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_times))

    for i in range(n_times):
        t = data['spectra_times'][i]
        spectrum = data['spectra'][i]
        ax.loglog(k_bins, spectrum, '-', color=colors[i], linewidth=2,
                  label=f't = {t:.1f} τ_A', alpha=0.7)

    # Average spectrum
    ax.loglog(k_bins, avg_spectrum, 'k-', linewidth=3,
              label='Time average', alpha=0.8)

    ax.set_xlabel(r'$k_\perp$')
    ax.set_ylabel(r'$E(k_\perp)$')
    ax.set_title(f'Spectrum Evolution in Steady State (N={N}³)')
    ax.legend(loc='upper right', ncol=2)
    ax.set_xlim([1, N//3])
    ax.set_ylim([1e-6, 1])
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(output_dir / "turbulent_cascade_evolution.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: turbulent_cascade_evolution.pdf")

    # ==========================================================================
    # Figure 5: Compensated spectrum E(k) * k^(5/3)
    # ==========================================================================
    print("\nGenerating Figure 5: Compensated spectrum...")

    fig, ax = plt.subplots(figsize=(10, 7))

    # Compensated spectrum
    compensated = avg_spectrum * k_bins**(5/3)

    ax.semilogx(k_bins, compensated, 'b-', linewidth=2, label=f'N={N}³ simulation')

    # Mark inertial range
    inertial = (k_bins >= k_fit_min) & (k_bins <= k_fit_max)
    ax.semilogx(k_bins[inertial], compensated[inertial], 'r-', linewidth=3,
                label='Inertial range')

    # Show expected flat region for perfect k^(-5/3)
    avg_compensated = np.mean(compensated[inertial])
    ax.axhline(avg_compensated, color='k', linestyle='--', alpha=0.5,
               label=f'Mean = {avg_compensated:.3f}')

    # Add shaded regions
    ax.axvspan(2, 5, alpha=0.2, color='green', label='Forcing')
    ax.axvspan(k_fit_max, N//3, alpha=0.2, color='red', label='Dissipation')

    ax.set_xlabel(r'$k_\perp$')
    ax.set_ylabel(r'$E(k_\perp) \cdot k_\perp^{5/3}$')
    ax.set_title(f'Compensated Spectrum (N={N}³)')
    ax.legend(loc='best')
    ax.set_xlim([1, N//3])
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(output_dir / "turbulent_cascade_compensated.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: turbulent_cascade_compensated.pdf")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

    print(f"\nResolution: N={N}³")
    if alpha is not None:
        print(f"Spectral index: α = {alpha:.3f} ± {std_err:.3f}")
        print(f"Expected (Kolmogorov): α = 5/3 = 1.667")
        print(f"Deviation: {abs(alpha - 5/3):.3f} ({abs(alpha - 5/3)/(5/3)*100:.1f}%)")

    print(f"\nGenerated 5 publication figures:")
    print("  1. turbulent_cascade_spectrum_fit.pdf - Main result with fit")
    print("  2. turbulent_cascade_energy_balance.pdf - Energy balance")
    if data_64 is not None and data_128 is not None:
        print("  3. turbulent_cascade_convergence.pdf - Resolution study")
    print("  4. turbulent_cascade_evolution.pdf - Time evolution")
    print("  5. turbulent_cascade_compensated.pdf - Compensated spectrum")

    return alpha, std_err


if __name__ == "__main__":
    alpha, err = generate_figures()
    print(f"\n✓ Analysis complete!")