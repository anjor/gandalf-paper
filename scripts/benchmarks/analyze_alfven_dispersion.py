#!/usr/bin/env -S uv run python
"""
Analyze Alfvén Wave Dispersion Benchmark Results

Loads benchmark data, computes convergence rates, and generates publication-quality
figures for the GANDALF paper Verification section.

Usage:
    uv run scripts/benchmarks/analyze_alfven_dispersion.py

Output:
    paper/figures/alfven_energy_evolution.pdf
    paper/figures/alfven_frequency_convergence.pdf
    paper/figures/alfven_phase_accuracy.pdf
    paper/figures/alfven_dispersion_validation.pdf

Physics:
    Validates that GANDALF correctly reproduces the Alfvén wave dispersion relation
    ω = k∥v_A with quantified convergence rates:
    - Spatial: Exponential convergence (spectral accuracy)
    - Temporal: O(Δt²) convergence (RK2/GANDALF integrating factor)

References:
    - Schekochihin et al. (2009) ApJS 182:310 - KRMHD formulation
    - Numata et al. (2010) PoP 17:102316 - Spectral accuracy benchmarks
"""

import sys
from pathlib import Path
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Dict, List, Tuple
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
    omega_measured: float
    omega_analytical: float
    relative_error: float
    k_parallel: float
    times: np.ndarray
    energy_history: np.ndarray
    phase_history: np.ndarray
    amplitude_history: np.ndarray


def load_benchmark_file(filepath: Path) -> BenchmarkResult:
    """Load a single benchmark HDF5 file."""
    with h5py.File(filepath, 'r') as f:
        return BenchmarkResult(
            resolution=int(f.attrs['resolution']),
            dt=float(f.attrs['dt']),
            omega_measured=float(f['omega_measured'][()]),
            omega_analytical=float(f['omega_analytical'][()]),
            relative_error=float(f['relative_error'][()]),
            k_parallel=float(f['k_parallel'][()]),
            times=np.array(f['times'][:]),
            energy_history=np.array(f['energy_history'][:]),
            phase_history=np.array(f['phase_history'][:]),
            amplitude_history=np.array(f['amplitude_history'][:]),
        )


def load_spatial_convergence(data_dir: Path) -> List[BenchmarkResult]:
    """Load all spatial convergence results, sorted by resolution."""
    results = []
    spatial_dir = data_dir / 'spatial_convergence'

    for filepath in sorted(spatial_dir.glob('N*.h5')):
        results.append(load_benchmark_file(filepath))

    # Sort by resolution
    results.sort(key=lambda r: r.resolution)
    return results


def load_temporal_convergence(data_dir: Path) -> List[BenchmarkResult]:
    """Load all temporal convergence results, sorted by timestep."""
    results = []
    temporal_dir = data_dir / 'temporal_convergence'

    for filepath in sorted(temporal_dir.glob('dt_*.h5')):
        results.append(load_benchmark_file(filepath))

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
    # Fit in log space: log(E) = log(A) - α*N
    log_errors = np.log(errors)
    coeffs = np.polyfit(resolutions, log_errors, deg=1)
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
    # Fit in log-log space: log(E) = log(C) + p*log(dt)
    log_dts = np.log(dts)
    log_errors = np.log(errors)
    coeffs = np.polyfit(log_dts, log_errors, deg=1)
    p = coeffs[0]  # Slope = exponent
    C = np.exp(coeffs[1])  # Exponential of intercept
    return p, C


def plot_energy_evolution(results: List[BenchmarkResult], output_dir: Path):
    """
    Figure 1: Energy vs time showing clean oscillation.

    Uses highest resolution result to show best accuracy.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    # Use highest resolution result
    result = results[-1]

    # Plot energy evolution
    ax.plot(result.times, result.energy_history, 'k-', linewidth=1.2, label='Total energy')

    # Mark initial and final energy
    E_0 = result.energy_history[0]
    E_f = result.energy_history[-1]
    rel_change = abs(E_f - E_0) / E_0

    ax.axhline(E_0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, label='Initial energy')

    # Labels and formatting
    ax.set_xlabel(r'Time $t$ $(v_A/L)$')
    ax.set_ylabel(r'Total Energy $E$')
    ax.set_title(f'Alfvén Wave Energy Conservation ($N={result.resolution}^3$, $\\Delta t={result.dt}$)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    # Add text box with conservation info
    textstr = f'$\\Delta E / E_0 = {rel_change:.2e}$'
    ax.text(0.95, 0.05, textstr, transform=ax.transAxes,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.tight_layout()
    output_file = output_dir / 'alfven_energy_evolution.pdf'
    plt.savefig(output_file, format='pdf', dpi=300)
    plt.close()
    print(f"✓ Generated: {output_file}")


def plot_frequency_convergence(spatial_results: List[BenchmarkResult],
                               temporal_results: List[BenchmarkResult],
                               output_dir: Path):
    """
    Figure 2: Two-panel convergence plot.
    (a) Spatial convergence: error vs N (semi-log)
    (b) Temporal convergence: error vs Δt (log-log)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # ===== Panel (a): Spatial Convergence =====
    resolutions = np.array([r.resolution for r in spatial_results])
    errors_spatial = np.array([r.relative_error for r in spatial_results])

    # Fit exponential convergence
    alpha, A = fit_exponential_convergence(resolutions, errors_spatial)

    # Plot data
    ax1.semilogy(resolutions, errors_spatial, 'o-', color='C0',
                 label='Measured error', markersize=7, linewidth=1.5)

    # Plot fit
    N_fit = np.linspace(resolutions[0], resolutions[-1], 100)
    E_fit = A * np.exp(-alpha * N_fit)
    ax1.semilogy(N_fit, E_fit, '--', color='C1',
                label=f'Fit: $E = A e^{{-\\alpha N}}$\n$\\alpha = {alpha:.3f}$', linewidth=1.2)

    ax1.set_xlabel(r'Resolution $N$ ($N^3$ grid)')
    ax1.set_ylabel(r'Relative frequency error $|\omega - \omega_\mathrm{analytical}| / \omega_\mathrm{analytical}$')
    ax1.set_title('(a) Spatial Convergence (Spectral Method)')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(loc='best')

    # ===== Panel (b): Temporal Convergence =====
    dts = np.array([r.dt for r in temporal_results])
    errors_temporal = np.array([r.relative_error for r in temporal_results])

    # Fit power law convergence
    p, C = fit_power_law_convergence(dts, errors_temporal)

    # Plot data
    ax2.loglog(dts, errors_temporal, 's-', color='C2',
              label='Measured error', markersize=7, linewidth=1.5)

    # Plot fit
    dt_fit = np.linspace(dts[-1], dts[0], 100)
    E_fit_temp = C * dt_fit**p
    ax2.loglog(dt_fit, E_fit_temp, '--', color='C3',
              label=f'Fit: $E = C\\,\\Delta t^p$\n$p = {p:.2f}$', linewidth=1.2)

    # Reference O(Δt²) line
    if len(dts) > 1:
        dt_ref = np.array([dts[-1], dts[0]])
        E_ref = errors_temporal[-1] * (dt_ref / dts[-1])**2
        ax2.loglog(dt_ref, E_ref, ':', color='gray', label=r'$O(\Delta t^2)$ reference', linewidth=1.5)

    ax2.set_xlabel(r'Timestep $\Delta t$')
    ax2.set_ylabel(r'Relative frequency error')
    ax2.set_title('(b) Temporal Convergence (GANDALF + RK2)')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(loc='best')

    plt.tight_layout()
    output_file = output_dir / 'alfven_frequency_convergence.pdf'
    plt.savefig(output_file, format='pdf', dpi=300)
    plt.close()
    print(f"✓ Generated: {output_file}")

    # Print convergence summary
    print(f"\n  Spatial convergence: α = {alpha:.3f} (exponential)")
    print(f"  Temporal convergence: p = {p:.2f} (power law, expect ~2.0 for RK2)")


def plot_phase_accuracy(results: List[BenchmarkResult], output_dir: Path):
    """
    Figure 3: Phase error accumulation over time for different resolutions.
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(results)))

    for result, color in zip(results, colors):
        # Compute phase error: difference from analytical phase
        omega_analytical = result.omega_analytical
        phase_analytical = omega_analytical * result.times
        phase_error = result.phase_history - phase_analytical

        # Plot
        ax.plot(result.times, np.abs(phase_error),
               label=f'$N={result.resolution}^3$', color=color, linewidth=1.3)

    ax.set_xlabel(r'Time $t$ $(v_A/L)$')
    ax.set_ylabel(r'Absolute phase error $|\phi - \omega_\mathrm{analytical} \cdot t|$ (rad)')
    ax.set_title('Phase Error Accumulation')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='best', ncol=2)

    plt.tight_layout()
    output_file = output_dir / 'alfven_phase_accuracy.pdf'
    plt.savefig(output_file, format='pdf', dpi=300)
    plt.close()
    print(f"✓ Generated: {output_file}")


def plot_spatial_convergence_only(spatial_results: List[BenchmarkResult], output_dir: Path):
    """
    Generate spatial convergence plot only (when temporal data not available).
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    resolutions = np.array([r.resolution for r in spatial_results])
    errors_spatial = np.array([r.relative_error for r in spatial_results])

    # Fit exponential convergence
    alpha, A = fit_exponential_convergence(resolutions, errors_spatial)

    # Plot data
    ax.semilogy(resolutions, errors_spatial, 'o-', color='C0',
                label='Measured error', markersize=8, linewidth=2)

    # Plot fit
    N_fit = np.linspace(resolutions[0], resolutions[-1], 100)
    E_fit = A * np.exp(-alpha * N_fit)
    ax.semilogy(N_fit, E_fit, '--', color='C1',
                label=f'Fit: $E = A e^{{-\\alpha N}}$\n$\\alpha = {alpha:.3f}$', linewidth=1.5)

    ax.set_xlabel(r'Resolution $N$ ($N^3$ grid)')
    ax.set_ylabel(r'Relative frequency error')
    ax.set_title('Spatial Convergence (Spectral Method)')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='best')

    plt.tight_layout()
    output_file = output_dir / 'alfven_spatial_convergence.pdf'
    plt.savefig(output_file, format='pdf', dpi=300)
    plt.close()
    print(f"✓ Generated: {output_file}")
    print(f"  Spatial convergence rate: α = {alpha:.3f} (exponential)")


def plot_dispersion_validation(spatial_results: List[BenchmarkResult],
                               temporal_results: List[BenchmarkResult],
                               output_dir: Path):
    """
    Figure 4: Measured ω vs analytical ω for all runs.
    Shows all data points lie on the ω_measured = ω_analytical line.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # Collect all results
    all_results = spatial_results + temporal_results

    omega_analytical = np.array([r.omega_analytical for r in all_results])
    omega_measured = np.array([r.omega_measured for r in all_results])

    # Separate spatial and temporal for different markers
    omega_ana_spatial = np.array([r.omega_analytical for r in spatial_results])
    omega_meas_spatial = np.array([r.omega_measured for r in spatial_results])
    omega_ana_temporal = np.array([r.omega_analytical for r in temporal_results])
    omega_meas_temporal = np.array([r.omega_measured for r in temporal_results])

    # Plot perfect agreement line
    omega_range = [omega_analytical.min() * 0.99, omega_analytical.max() * 1.01]
    ax.plot(omega_range, omega_range, 'k--', linewidth=1.5, label='Perfect agreement', alpha=0.5)

    # Plot data
    ax.plot(omega_ana_spatial, omega_meas_spatial, 'o',
           color='C0', markersize=8, label='Spatial convergence', markeredgecolor='white', markeredgewidth=0.5)
    ax.plot(omega_ana_temporal, omega_meas_temporal, 's',
           color='C2', markersize=8, label='Temporal convergence', markeredgecolor='white', markeredgewidth=0.5)

    ax.set_xlabel(r'Analytical frequency $\omega_\mathrm{analytical} = k_\parallel v_A$')
    ax.set_ylabel(r'Measured frequency $\omega_\mathrm{measured}$')
    ax.set_title('Alfvén Wave Dispersion Relation Validation')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    ax.set_aspect('equal', adjustable='box')

    # Add error statistics text box
    max_error = np.max([r.relative_error for r in all_results])
    mean_error = np.mean([r.relative_error for r in all_results])
    textstr = f'Max relative error: ${max_error:.2e}$\nMean relative error: ${mean_error:.2e}$'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
           verticalalignment='top', horizontalalignment='left',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

    plt.tight_layout()
    output_file = output_dir / 'alfven_dispersion_validation.pdf'
    plt.savefig(output_file, format='pdf', dpi=300)
    plt.close()
    print(f"✓ Generated: {output_file}")

    print(f"\n  Maximum relative error: {max_error:.2e}")
    print(f"  Mean relative error: {mean_error:.2e}")


def main():
    """Main analysis routine."""
    # Paths
    script_dir = Path(__file__).resolve().parent
    paper_dir = script_dir.parent.parent
    data_dir = paper_dir / 'data' / 'benchmarks' / 'alfven_wave'
    output_dir = paper_dir / 'paper' / 'figures'

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("ALFVÉN WAVE DISPERSION BENCHMARK ANALYSIS")
    print("="*70)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Load results
    print("Loading benchmark results...")
    spatial_results = load_spatial_convergence(data_dir)
    temporal_results = load_temporal_convergence(data_dir)

    print(f"  Spatial convergence: {len(spatial_results)} resolutions")
    print(f"  Temporal convergence: {len(temporal_results)} timesteps")
    print()

    # Generate figures
    print("Generating publication-quality figures...")
    print()

    plot_energy_evolution(spatial_results, output_dir)

    if len(temporal_results) > 0:
        plot_frequency_convergence(spatial_results, temporal_results, output_dir)
        plot_dispersion_validation(spatial_results, temporal_results, output_dir)
    else:
        print("⚠️  Skipping temporal convergence plots (no temporal data yet)")
        print("⚠️  Generating spatial-only plots...")
        plot_spatial_convergence_only(spatial_results, output_dir)

    plot_phase_accuracy(spatial_results, output_dir)

    print()
    print("="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nGenerated {4} publication-quality figures in: {output_dir}")
    print("\nNext steps:")
    print("  1. Review figures: ls -lh paper/figures/alfven_*.pdf")
    print("  2. Write verification.tex section")
    print("  3. Compile paper and verify figure rendering")
    print()


if __name__ == '__main__':
    main()
