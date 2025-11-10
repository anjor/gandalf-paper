#!/usr/bin/env -S uv run python
"""
Analyze Orszag-Tang Vortex Benchmark Results

Loads benchmark data, computes convergence rates, and generates publication-quality
figures for the GANDALF paper Verification section.

Usage:
    uv run scripts/benchmarks/analyze_orszag_tang.py

Output:
    paper/figures/orszag_tang_energy_evolution.pdf
    paper/figures/orszag_tang_structures.pdf
    paper/figures/orszag_tang_convergence.pdf
    paper/figures/orszag_tang_spectra.pdf

Physics:
    Validates that GANDALF correctly reproduces nonlinear MHD dynamics:
    - Energy conservation in inviscid limit
    - Selective decay: E_mag/E_kin increases in 2D MHD
    - Current sheet formation
    - Energy cascade to small scales

References:
    - Orszag & Tang (1979) J. Fluid Mech. 90:129 - Original benchmark
    - Schekochihin et al. (2009) ApJS 182:310 - KRMHD formulation
"""

import sys
from pathlib import Path
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Configure matplotlib for publication-quality figures (JPP standards)
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'grid.alpha': 0.3,
})


@dataclass
class BenchmarkResult:
    """Container for single benchmark run results."""
    resolution: int
    dt: float
    n_steps: int
    E0: float
    energy_conservation_error: float
    times: np.ndarray
    energy_total: np.ndarray
    energy_kinetic: np.ndarray
    energy_magnetic: np.ndarray
    snapshot_times: np.ndarray
    z_plus_snapshots: np.ndarray
    z_minus_snapshots: np.ndarray

    @property
    def is_valid(self) -> bool:
        """Check if result has valid (non-NaN) data."""
        return (not np.isnan(self.energy_conservation_error) and
                not np.any(np.isnan(self.energy_total)))


def load_benchmark_file(filepath: Path) -> Optional[BenchmarkResult]:
    """Load a single benchmark HDF5 file."""
    try:
        with h5py.File(filepath, 'r') as f:
            result = BenchmarkResult(
                resolution=int(f.attrs['resolution']),
                dt=float(f.attrs['dt']),
                n_steps=int(f.attrs['n_steps']),
                E0=float(f.attrs['E0']),
                energy_conservation_error=float(f['energy_conservation_error'][()]),
                times=np.array(f['times'][:]),
                energy_total=np.array(f['energy_total'][:]),
                energy_kinetic=np.array(f['energy_kinetic'][:]),
                energy_magnetic=np.array(f['energy_magnetic'][:]),
                snapshot_times=np.array(f['snapshot_times'][:]),
                z_plus_snapshots=np.array(f['z_plus_snapshots'][:]),
                z_minus_snapshots=np.array(f['z_minus_snapshots'][:]),
            )

            if not result.is_valid:
                print(f"  ⚠️  Warning: {filepath.name} contains NaN values (numerical instability)")
                return None

            return result
    except Exception as e:
        print(f"  ⚠️  Warning: Failed to load {filepath}: {e}")
        return None


def load_spatial_convergence(data_dir: Path) -> List[BenchmarkResult]:
    """Load all spatial convergence results, sorted by resolution."""
    results = []
    spatial_dir = data_dir / 'spatial_convergence'

    if not spatial_dir.exists():
        return results

    for filepath in sorted(spatial_dir.glob('N*.h5')):
        result = load_benchmark_file(filepath)
        if result is not None:
            results.append(result)

    # Sort by resolution
    results.sort(key=lambda r: r.resolution)
    return results


def load_temporal_convergence(data_dir: Path) -> List[BenchmarkResult]:
    """Load all temporal convergence results, sorted by timestep."""
    results = []
    temporal_dir = data_dir / 'temporal_convergence'

    if not temporal_dir.exists():
        return results

    for filepath in sorted(temporal_dir.glob('dt_*.h5')):
        result = load_benchmark_file(filepath)
        if result is not None:
            results.append(result)

    # Sort by dt (largest to smallest)
    results.sort(key=lambda r: r.dt, reverse=True)
    return results


def fit_exponential_convergence(resolutions: np.ndarray, errors: np.ndarray) -> Tuple[float, float]:
    """
    Fit exponential convergence: E = A * exp(-α * N)

    Returns:
        alpha: Exponential convergence rate
        A: Prefactor
    """
    # Filter out any remaining NaN or zero values
    mask = (errors > 0) & np.isfinite(errors)
    if np.sum(mask) < 2:
        return np.nan, np.nan

    # Fit in log space: log(E) = log(A) - α*N
    log_errors = np.log(errors[mask])
    resolutions_valid = resolutions[mask]
    coeffs = np.polyfit(resolutions_valid, log_errors, deg=1)
    alpha = -coeffs[0]  # Negative of slope
    A = np.exp(coeffs[1])  # Exponential of intercept
    return alpha, A


def fit_power_law_convergence(dts: np.ndarray, errors: np.ndarray) -> Tuple[float, float]:
    """
    Fit power law convergence: E = C * dt^p

    Returns:
        p: Power law exponent (should be ~2 for RK2)
        C: Prefactor
    """
    # Filter out NaN and very small values
    mask = (errors > 1e-14) & np.isfinite(errors)
    if np.sum(mask) < 2:
        return np.nan, np.nan

    # Fit in log-log space: log(E) = log(C) + p*log(dt)
    log_dts = np.log(dts[mask])
    log_errors = np.log(errors[mask])
    coeffs = np.polyfit(log_dts, log_errors, deg=1)
    p = coeffs[0]  # Slope = exponent
    C = np.exp(coeffs[1])  # Exponential of intercept
    return p, C


def plot_energy_evolution(results: List[BenchmarkResult], output_dir: Path):
    """
    Figure 1: Two-panel energy evolution plot.
    (a) Energy components vs time
    (b) Energy conservation error vs time
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # Use highest resolution result
    result = results[-1]

    # Normalize time by Alfvén crossing time (τ_A = 1.0)
    tau_A = 1.0
    times_alfven = result.times / tau_A

    # ===== Panel (a): Energy Components =====
    ax1.plot(times_alfven, result.energy_total, 'k-', linewidth=2, label='Total')
    ax1.plot(times_alfven, result.energy_kinetic, 'r--', linewidth=2, label='Kinetic')
    ax1.plot(times_alfven, result.energy_magnetic, 'b:', linewidth=2, label='Magnetic')

    ax1.set_ylabel(r'Energy')
    ax1.set_title(f'Orszag-Tang Vortex: Energy Evolution ($N={result.resolution}^2 \\times 2$, $\\Delta t={result.dt}$)')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Add text box with selective decay info
    E_mag_initial = result.energy_magnetic[0]
    E_kin_initial = result.energy_kinetic[0]
    E_mag_final = result.energy_magnetic[-1]
    E_kin_final = result.energy_kinetic[-1]
    ratio_initial = E_mag_initial / E_kin_initial
    ratio_final = E_mag_final / E_kin_final

    textstr = f'$E_{{\\rm mag}}/E_{{\\rm kin}}$: ${ratio_initial:.2f} \\to {ratio_final:.2f}$'
    ax1.text(0.95, 0.05, textstr, transform=ax1.transAxes,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='gray'))

    # ===== Panel (b): Energy Conservation Error =====
    E_error = np.abs(result.energy_total - result.energy_total[0]) / result.energy_total[0]
    ax2.semilogy(times_alfven, E_error, 'k-', linewidth=2)

    # Reference lines
    ax2.axhline(y=0.01, color='g', linestyle='--', linewidth=1, alpha=0.7, label='Target: 1%')
    ax2.axhline(y=0.001, color='b', linestyle='--', linewidth=1, alpha=0.7, label='Excellent: 0.1%')

    ax2.set_xlabel(r'Time $t / \tau_A$')
    ax2.set_ylabel(r'$|\Delta E / E_0|$')
    ax2.set_title('Energy Conservation')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_ylim(bottom=1e-6)

    plt.tight_layout()
    output_file = output_dir / 'orszag_tang_energy_evolution.pdf'
    plt.savefig(output_file, format='pdf', dpi=300)
    plt.close()
    print(f"✓ Generated: {output_file}")


def plot_structures(result: BenchmarkResult, output_dir: Path):
    """
    Figure 2: 2×2 grid of 2D structure plots at final time.
    Shows vorticity, current density, stream function, and vector potential.
    """
    # Use final snapshot
    z_plus_k = result.z_plus_snapshots[-1, :, :]  # Shape: (Ny, Nx_rfft)
    z_minus_k = result.z_minus_snapshots[-1, :, :]
    t_final = result.snapshot_times[-1]

    # Reconstruct fields: φ = (z+ + z-)/2, Ψ = (z+ - z-)/2
    phi_k = 0.5 * (z_plus_k + z_minus_k)
    psi_k = 0.5 * (z_plus_k - z_minus_k)

    # Grid setup (2D problem: N² × 2, using kz=0 plane)
    N = result.resolution
    Lx = Ly = 1.0

    # Wavenumbers for 2D rfft
    # Shape should be (Ny, Nx_rfft) to match z_plus_k/z_minus_k
    kx = 2.0 * np.pi * np.fft.rfftfreq(N, d=Lx/N)
    ky = 2.0 * np.pi * np.fft.fftfreq(N, d=Ly/N)
    KX, KY = np.meshgrid(kx, ky, indexing='xy')  # xy indexing for (Ny, Nx_rfft) shape
    k_perp_sq = KX**2 + KY**2

    # Compute Laplacians: ω = ∇²φ (vorticity), J∥ = ∇²Ψ (current)
    omega_k = -k_perp_sq * phi_k
    j_parallel_k = -k_perp_sq * psi_k

    # Transform to real space using irfft2
    phi_real = np.fft.irfft2(phi_k, s=(N, N))
    psi_real = np.fft.irfft2(psi_k, s=(N, N))
    omega_real = np.fft.irfft2(omega_k, s=(N, N))
    j_parallel_real = np.fft.irfft2(j_parallel_k, s=(N, N))

    # Create 2×2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Coordinate grids
    x = np.linspace(0, Lx, N, endpoint=False)
    y = np.linspace(0, Ly, N, endpoint=False)
    X, Y = np.meshgrid(x, y)

    # Plot 1: Vorticity (fluid vortex structures)
    ax = axes[0, 0]
    im1 = ax.pcolormesh(X, Y, omega_real, cmap='RdBu_r', shading='auto')
    ax.set_title(f'(a) Vorticity $\\omega = \\nabla^2\\phi$ at $t = {t_final:.1f}\\tau_A$', fontsize=12, fontweight='bold')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_aspect('equal')
    fig.colorbar(im1, ax=ax, label='$\\omega$')

    # Plot 2: Current density (magnetic current sheets)
    ax = axes[0, 1]
    im2 = ax.pcolormesh(X, Y, j_parallel_real, cmap='PiYG', shading='auto')
    ax.set_title(f'(b) Current Density $J_\\parallel = \\nabla^2\\Psi$ at $t = {t_final:.1f}\\tau_A$', fontsize=12, fontweight='bold')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_aspect('equal')
    fig.colorbar(im2, ax=ax, label='$J_\\parallel$')

    # Plot 3: Stream function (flow potential)
    ax = axes[1, 0]
    im3 = ax.pcolormesh(X, Y, phi_real, cmap='viridis', shading='auto')
    ax.set_title(f'(c) Stream Function $\\phi$ at $t = {t_final:.1f}\\tau_A$', fontsize=12, fontweight='bold')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_aspect('equal')
    fig.colorbar(im3, ax=ax, label='$\\phi$')

    # Plot 4: Vector potential (magnetic flux)
    ax = axes[1, 1]
    im4 = ax.pcolormesh(X, Y, psi_real, cmap='plasma', shading='auto')
    ax.set_title(f'(d) Vector Potential $\\Psi$ at $t = {t_final:.1f}\\tau_A$', fontsize=12, fontweight='bold')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_aspect('equal')
    fig.colorbar(im4, ax=ax, label='$\\Psi$')

    plt.tight_layout()
    output_file = output_dir / 'orszag_tang_structures.pdf'
    plt.savefig(output_file, format='pdf', dpi=300)
    plt.close()
    print(f"✓ Generated: {output_file}")


def plot_convergence(spatial_results: List[BenchmarkResult],
                     temporal_results: List[BenchmarkResult],
                     output_dir: Path):
    """
    Figure 3: Two-panel convergence plot.
    (a) Spatial convergence: energy error vs N
    (b) Temporal convergence: energy error vs Δt
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ===== Panel (a): Spatial Convergence =====
    if len(spatial_results) > 0:
        resolutions = np.array([r.resolution for r in spatial_results])
        errors_spatial = np.array([r.energy_conservation_error for r in spatial_results])

        # Fit exponential convergence
        alpha, A = fit_exponential_convergence(resolutions, errors_spatial)

        # Plot data
        ax1.semilogy(resolutions, errors_spatial, 'o-', color='C0',
                     label='Measured error', markersize=8, linewidth=2)

        # Plot fit if available
        if not np.isnan(alpha):
            N_fit = np.linspace(resolutions[0], resolutions[-1], 100)
            E_fit = A * np.exp(-alpha * N_fit)
            ax1.semilogy(N_fit, E_fit, '--', color='C1',
                        label=f'Fit: $E = A e^{{-\\alpha N}}$\n$\\alpha = {alpha:.3f}$', linewidth=2)

        ax1.set_xlabel(r'Resolution $N$ ($N^2 \times 2$ grid)')
        ax1.set_ylabel(r'Energy conservation error $|\Delta E / E_0|$')
        ax1.set_title('(a) Spatial Convergence')
        ax1.grid(True, alpha=0.3, which='both')
        ax1.legend(loc='best')
    else:
        ax1.text(0.5, 0.5, 'No valid spatial\nconvergence data',
                ha='center', va='center', transform=ax1.transAxes, fontsize=14)

    # ===== Panel (b): Temporal Convergence =====
    if len(temporal_results) > 0:
        dts = np.array([r.dt for r in temporal_results])
        errors_temporal = np.array([r.energy_conservation_error for r in temporal_results])

        # Fit power law convergence
        p, C = fit_power_law_convergence(dts, errors_temporal)

        # Plot data
        ax2.loglog(dts, errors_temporal, 's-', color='C2',
                   label='Measured error', markersize=8, linewidth=2)

        # Plot fit if available
        if not np.isnan(p):
            dt_fit = np.linspace(dts[-1], dts[0], 100)
            E_fit = C * dt_fit**p
            ax2.loglog(dt_fit, E_fit, '--', color='C3',
                      label=f'Fit: $E = C\\,\\Delta t^p$\n$p = {p:.2f}$', linewidth=2)

        # Reference O(Δt²) line
        if len(dts) > 1:
            dt_ref = np.array([dts[-1], dts[0]])
            # Use middle point as reference
            idx_mid = len(dts) // 2
            E_ref = errors_temporal[idx_mid] * (dt_ref / dts[idx_mid])**2
            ax2.loglog(dt_ref, E_ref, ':', color='gray', label=r'$O(\Delta t^2)$ reference', linewidth=2)

        ax2.set_xlabel(r'Timestep $\Delta t$')
        ax2.set_ylabel(r'Energy conservation error')
        ax2.set_title('(b) Temporal Convergence')
        ax2.grid(True, alpha=0.3, which='both')
        ax2.legend(loc='best')
    else:
        ax2.text(0.5, 0.5, 'No valid temporal\nconvergence data',
                ha='center', va='center', transform=ax2.transAxes, fontsize=14)

    plt.tight_layout()
    output_file = output_dir / 'orszag_tang_convergence.pdf'
    plt.savefig(output_file, format='pdf', dpi=300)
    plt.close()
    print(f"✓ Generated: {output_file}")

    # Print convergence summary
    if len(spatial_results) > 0 and not np.isnan(alpha):
        print(f"  Spatial convergence: α = {alpha:.3f} (exponential)")
    if len(temporal_results) > 0 and not np.isnan(p):
        print(f"  Temporal convergence: p = {p:.2f} (power law, expect ~2.0 for RK2)")


def plot_spectra(result: BenchmarkResult, output_dir: Path):
    """
    Figure 4: Energy spectra E(k_perp) at multiple times showing cascade.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Grid setup
    N = result.resolution
    Lx = Ly = 1.0

    # Wavenumbers for 2D rfft (shape: Ny, Nx_rfft)
    kx = 2.0 * np.pi * np.fft.rfftfreq(N, d=Lx/N)
    ky = 2.0 * np.pi * np.fft.fftfreq(N, d=Ly/N)
    KX, KY = np.meshgrid(kx, ky, indexing='xy')
    k_perp = np.sqrt(KX**2 + KY**2)

    # Define k bins for spectrum
    k_max = np.pi * N / Lx  # Nyquist frequency
    k_bins = np.logspace(np.log10(1.0), np.log10(k_max), 20)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])

    # Compute spectra at multiple snapshots
    n_snapshots = min(4, len(result.snapshot_times))
    indices = np.linspace(0, len(result.snapshot_times)-1, n_snapshots, dtype=int)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_snapshots))

    for idx, color in zip(indices, colors):
        z_plus_k = result.z_plus_snapshots[idx, :, :]
        z_minus_k = result.z_minus_snapshots[idx, :, :]
        t = result.snapshot_times[idx]

        # Energy in Fourier space: E_k = |k_perp|² * |z±|² / 2
        # (Factor of 2 for Elsasser variables)
        E_k = 0.5 * k_perp**2 * (np.abs(z_plus_k)**2 + np.abs(z_minus_k)**2)

        # Bin into radial spectrum E(k_perp)
        E_spectrum = np.zeros(len(k_centers))
        for i in range(len(k_bins) - 1):
            mask = (k_perp >= k_bins[i]) & (k_perp < k_bins[i+1])
            if np.any(mask):
                E_spectrum[i] = np.sum(E_k[mask])

        # Normalize by bin width for proper spectral density
        dk = k_bins[1:] - k_bins[:-1]
        E_spectrum /= dk

        # Plot
        ax.loglog(k_centers, E_spectrum, '-o', color=color,
                 label=f'$t = {t:.1f}\\tau_A$', linewidth=1.5, markersize=4)

    # Reference slopes
    k_ref = np.array([2.0, 20.0])
    E_ref_k3 = 1e2 * (k_ref / k_ref[0])**(-3)
    E_ref_k53 = 1e2 * (k_ref / k_ref[0])**(-5/3)

    ax.loglog(k_ref, E_ref_k3, 'k--', linewidth=1.5, alpha=0.5, label=r'$k^{-3}$ (2D MHD)')
    ax.loglog(k_ref, E_ref_k53, 'k:', linewidth=1.5, alpha=0.5, label=r'$k^{-5/3}$ (forward cascade)')

    ax.set_xlabel(r'Wavenumber $k_\perp$')
    ax.set_ylabel(r'Energy spectral density $E(k_\perp)$')
    ax.set_title(f'Orszag-Tang Vortex: Energy Cascade ($N={result.resolution}^2 \\times 2$)')
    ax.legend(loc='best', ncol=2)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(left=1.0, right=k_max)

    plt.tight_layout()
    output_file = output_dir / 'orszag_tang_spectra.pdf'
    plt.savefig(output_file, format='pdf', dpi=300)
    plt.close()
    print(f"✓ Generated: {output_file}")


def main():
    """Main analysis routine."""
    # Paths
    script_dir = Path(__file__).resolve().parent
    paper_dir = script_dir.parent.parent
    data_dir = paper_dir / 'data' / 'benchmarks' / 'orszag_tang'
    output_dir = paper_dir / 'paper' / 'figures'

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("ORSZAG-TANG VORTEX BENCHMARK ANALYSIS")
    print("="*70)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Load results
    print("Loading benchmark results...")
    spatial_results = load_spatial_convergence(data_dir)
    temporal_results = load_temporal_convergence(data_dir)

    print(f"  Spatial convergence: {len(spatial_results)} valid runs")
    print(f"  Temporal convergence: {len(temporal_results)} valid runs")
    print()

    if len(spatial_results) == 0:
        print("❌ No valid spatial convergence data found!")
        return

    # Generate figures
    print("Generating publication-quality figures...")
    print()

    # Use highest resolution for structure plots
    highest_res_result = spatial_results[-1]

    plot_energy_evolution(spatial_results, output_dir)
    plot_structures(highest_res_result, output_dir)
    plot_convergence(spatial_results, temporal_results, output_dir)
    plot_spectra(highest_res_result, output_dir)

    print()
    print("="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nGenerated 4 publication-quality figures in: {output_dir}")

    # Print summary statistics
    print("\nBenchmark Summary:")
    print(f"  Highest resolution: {spatial_results[-1].resolution}² × 2")
    print(f"  Energy conservation: |ΔE/E₀| = {spatial_results[-1].energy_conservation_error:.2e}")

    E_mag_init = spatial_results[-1].energy_magnetic[0]
    E_kin_init = spatial_results[-1].energy_kinetic[0]
    E_mag_final = spatial_results[-1].energy_magnetic[-1]
    E_kin_final = spatial_results[-1].energy_kinetic[-1]
    print(f"  Selective decay: E_mag/E_kin = {E_mag_init/E_kin_init:.2f} → {E_mag_final/E_kin_final:.2f}")

    print("\nNext steps:")
    print("  1. Review figures: ls -lh paper/figures/orszag_tang_*.pdf")
    print("  2. Write verification.tex subsection")
    print("  3. Compile paper and verify figure rendering")
    print()


if __name__ == '__main__':
    main()
