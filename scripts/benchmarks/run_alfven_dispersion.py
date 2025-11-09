#!/usr/bin/env -S uv run python
"""
Alfvén Wave Dispersion Benchmark for GANDALF Paper

Runs spatial and temporal convergence studies for linear Alfvén wave propagation.
Validates dispersion relation: ω = k∥v_A

This script generates benchmark data for the Verification section of the paper.

Usage:
    uv run scripts/benchmarks/run_alfven_dispersion.py --mode spatial
    uv run scripts/benchmarks/run_alfven_dispersion.py --mode temporal
    uv run scripts/benchmarks/run_alfven_dispersion.py --mode both

Output:
    data/benchmarks/alfven_wave/spatial_convergence/*.h5
    data/benchmarks/alfven_wave/temporal_convergence/*.h5

Physics:
    For a pure Alfvén wave with wavenumber k_z, the analytical dispersion relation
    predicts ω = k_z × v_A. We initialize a single Fourier mode and measure its
    frequency by tracking the phase evolution:

    ξ̂⁺(k_z, t) = A(t) exp(iφ(t))
    ω_measured = dφ/dt

    Convergence rates:
    - Spatial (spectral): Exponential convergence E ~ exp(-αN)
    - Temporal (RK2): Second-order convergence E ~ O(Δt²)

References:
    - Schekochihin et al. (2009) ApJS 182:310 - KRMHD formulation
    - Numata et al. (2010) PoP 17:102316 - AstroGK benchmarks
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import h5py
from typing import Dict, List, Tuple
import time

# Add GANDALF to path
gandalf_path = Path(__file__).resolve().parents[3] / "gandalf" / "src"
if str(gandalf_path) not in sys.path:
    sys.path.insert(0, str(gandalf_path))

import jax
import jax.numpy as jnp

from krmhd.physics import initialize_alfven_wave, KRMHDState, energy
from krmhd.spectral import SpectralGrid3D
from krmhd.timestepping import gandalf_step
from krmhd.diagnostics import EnergyHistory


def measure_alfven_frequency(
    grid: SpectralGrid3D,
    M: int,
    resolution: int,
    dt: float,
    n_periods: float = 5.0,
    kz_mode: int = 1,
    amplitude: float = 0.01,
    v_A: float = 1.0,
) -> Dict[str, float]:
    """
    Run single Alfvén wave simulation and measure frequency.

    Args:
        grid: Spectral grid
        M: Number of Hermite moments
        resolution: Grid resolution (N for N³ grid)
        dt: Time step
        n_periods: Number of Alfvén periods to integrate
        kz_mode: Parallel wavenumber mode (integer)
        amplitude: Wave amplitude (small for linear regime)
        v_A: Alfvén velocity

    Returns:
        Dictionary with:
            - omega_measured: Measured frequency from phase tracking
            - omega_analytical: Analytical ω = k_z × v_A
            - relative_error: |ω_measured - ω_analytical| / ω_analytical
            - times: Time array
            - energy_history: Total energy time series
            - phase_history: Phase evolution
            - amplitude_history: Amplitude evolution
            - k_parallel: Parallel wavenumber
    """
    print(f"  Running N={resolution}³, dt={dt:.6f}...")

    # Initialize Alfvén wave
    state = initialize_alfven_wave(
        grid=grid,
        M=M,
        kx_mode=0.0,
        ky_mode=0.0,
        kz_mode=float(kz_mode),
        amplitude=amplitude,
        v_th=1.0,
        beta_i=1.0,
        nu=0.0,
        Lambda=1.0,
    )

    # Analytical prediction
    k_parallel = grid.kz[kz_mode]
    omega_analytical = k_parallel * v_A
    T_period = 2.0 * np.pi / omega_analytical

    # Time integration parameters
    n_steps = int(n_periods * T_period / dt)

    # Storage for diagnostics
    times = []
    phases = []
    amplitudes = []
    energies = []

    # Find grid indices for the mode
    ikz = kz_mode
    iky = 0
    ikx = 0

    # Time integration loop
    for step in range(n_steps):
        # Record diagnostics
        times.append(float(state.time))

        # Extract Fourier mode amplitude (z_plus at k_z mode)
        mode_amplitude = state.z_plus[ikz, iky, ikx]
        amplitudes.append(float(jnp.abs(mode_amplitude)))
        phases.append(float(jnp.angle(mode_amplitude)))

        # Energy
        E_dict = energy(state)
        energies.append(float(E_dict['total']))

        # Evolve
        state = gandalf_step(state, dt=dt, eta=0.0, v_A=v_A)

    # Convert to arrays
    times = np.array(times)
    phases = np.array(phases)
    amplitudes = np.array(amplitudes)
    energies = np.array(energies)

    # Unwrap phase (handle 2π jumps)
    phases = np.unwrap(phases)

    # Measure frequency from phase evolution via linear fit
    # ω = dφ/dt, so fit φ(t) = ω*t + φ₀
    coeffs = np.polyfit(times, phases, deg=1)
    omega_measured = coeffs[0]  # Slope = frequency

    # Compute error
    relative_error = np.abs(omega_measured - omega_analytical) / omega_analytical

    print(f"    ω_analytical = {omega_analytical:.8f}")
    print(f"    ω_measured   = {omega_measured:.8f}")
    print(f"    Relative error = {relative_error:.2e}")
    print(f"    Energy conservation: ΔE/E = {(energies[-1] - energies[0]) / energies[0]:.2e}")

    return {
        'omega_measured': omega_measured,
        'omega_analytical': omega_analytical,
        'relative_error': relative_error,
        'times': times,
        'energy_history': energies,
        'phase_history': phases,
        'amplitude_history': amplitudes,
        'k_parallel': k_parallel,
        'resolution': resolution,
        'dt': dt,
        'n_periods': n_periods,
    }


def spatial_convergence_study(
    output_dir: Path,
    resolutions: List[int] = [32, 64, 128, 256],
    dt: float = 0.01,
    n_periods: float = 10.0,
    M: int = 20,
) -> None:
    """
    Run spatial convergence study: vary resolution, fixed timestep.

    Tests spectral method convergence (should be exponential).

    Args:
        output_dir: Directory to save results
        resolutions: List of grid resolutions (N for N³ grid)
        dt: Fixed timestep (well-resolved)
        n_periods: Number of periods to integrate
        M: Number of Hermite moments
    """
    print("\n" + "="*70)
    print("SPATIAL CONVERGENCE STUDY")
    print("="*70)
    print(f"Resolutions: {resolutions}")
    print(f"Fixed dt = {dt}")
    print(f"Integration time: {n_periods} Alfvén periods")
    print()

    output_dir.mkdir(parents=True, exist_ok=True)

    for N in resolutions:
        start_time = time.time()

        # Create grid
        Lx = Ly = Lz = 2.0 * np.pi
        grid = SpectralGrid3D.create(Nx=N, Ny=N, Nz=N, Lx=Lx, Ly=Ly, Lz=Lz)

        # Run simulation
        results = measure_alfven_frequency(
            grid=grid,
            M=M,
            resolution=N,
            dt=dt,
            n_periods=n_periods,
        )

        # Save to HDF5
        output_file = output_dir / f"N{N:03d}.h5"
        with h5py.File(output_file, 'w') as f:
            # Metadata
            f.attrs['resolution'] = N
            f.attrs['dt'] = dt
            f.attrs['n_periods'] = n_periods
            f.attrs['M'] = M
            f.attrs['study_type'] = 'spatial_convergence'

            # Results
            f.create_dataset('omega_measured', data=results['omega_measured'])
            f.create_dataset('omega_analytical', data=results['omega_analytical'])
            f.create_dataset('relative_error', data=results['relative_error'])
            f.create_dataset('k_parallel', data=results['k_parallel'])

            # Time series
            f.create_dataset('times', data=results['times'], compression='gzip')
            f.create_dataset('energy_history', data=results['energy_history'], compression='gzip')
            f.create_dataset('phase_history', data=results['phase_history'], compression='gzip')
            f.create_dataset('amplitude_history', data=results['amplitude_history'], compression='gzip')

        elapsed = time.time() - start_time
        print(f"  Saved to {output_file} ({elapsed:.1f}s)\n")

    print("Spatial convergence study complete!")


def temporal_convergence_study(
    output_dir: Path,
    timesteps: List[float] = None,
    resolution: int = 64,
    n_periods: float = 5.0,
    M: int = 20,
) -> None:
    """
    Run temporal convergence study: vary timestep, fixed resolution.

    Tests GANDALF integrating factor + RK2 convergence (should be O(Δt²)).

    Args:
        output_dir: Directory to save results
        timesteps: List of timesteps to test (if None, uses T/10, T/20, T/40, T/80, T/160)
        resolution: Fixed grid resolution
        n_periods: Number of periods to integrate
        M: Number of Hermite moments
    """
    print("\n" + "="*70)
    print("TEMPORAL CONVERGENCE STUDY")
    print("="*70)
    print(f"Fixed resolution: {resolution}³")
    print(f"Integration time: {n_periods} Alfvén periods")
    print()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create grid once (fixed resolution)
    Lx = Ly = Lz = 2.0 * np.pi
    grid = SpectralGrid3D.create(Nx=resolution, Ny=resolution, Nz=resolution, Lx=Lx, Ly=Ly, Lz=Lz)

    # Determine timesteps if not provided
    if timesteps is None:
        # Analytical period for kz=1, v_A=1: T = 2π/ω = 2π/1 = 2π
        T = 2.0 * np.pi
        timesteps = [T/10, T/20, T/40, T/80, T/160]

    print(f"Timesteps: {[f'{dt:.6f}' for dt in timesteps]}")
    print()

    for dt in timesteps:
        start_time = time.time()

        # Run simulation
        results = measure_alfven_frequency(
            grid=grid,
            M=M,
            resolution=resolution,
            dt=dt,
            n_periods=n_periods,
        )

        # Save to HDF5
        # Use scientific notation for filename to handle small dt
        dt_str = f"{dt:.6e}".replace('.', 'p').replace('+', '').replace('-', 'm')
        output_file = output_dir / f"dt_{dt_str}.h5"

        with h5py.File(output_file, 'w') as f:
            # Metadata
            f.attrs['resolution'] = resolution
            f.attrs['dt'] = dt
            f.attrs['n_periods'] = n_periods
            f.attrs['M'] = M
            f.attrs['study_type'] = 'temporal_convergence'

            # Results
            f.create_dataset('omega_measured', data=results['omega_measured'])
            f.create_dataset('omega_analytical', data=results['omega_analytical'])
            f.create_dataset('relative_error', data=results['relative_error'])
            f.create_dataset('k_parallel', data=results['k_parallel'])

            # Time series
            f.create_dataset('times', data=results['times'], compression='gzip')
            f.create_dataset('energy_history', data=results['energy_history'], compression='gzip')
            f.create_dataset('phase_history', data=results['phase_history'], compression='gzip')
            f.create_dataset('amplitude_history', data=results['amplitude_history'], compression='gzip')

        elapsed = time.time() - start_time
        print(f"  Saved to {output_file} ({elapsed:.1f}s)\n")

    print("Temporal convergence study complete!")


def main():
    """Main entry point for benchmark script."""
    parser = argparse.ArgumentParser(
        description='Run Alfvén wave dispersion benchmark for GANDALF paper',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['spatial', 'temporal', 'both'],
        default='both',
        help='Which convergence study to run (default: both)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(__file__).resolve().parents[2] / 'data' / 'benchmarks' / 'alfven_wave',
        help='Output directory for benchmark data'
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print("ALFVÉN WAVE DISPERSION BENCHMARK")
    print("="*70)
    print(f"Output directory: {args.output_dir}")
    print(f"JAX devices: {jax.devices()}")
    print("="*70)

    # Run requested studies
    if args.mode in ['spatial', 'both']:
        spatial_convergence_study(
            output_dir=args.output_dir / 'spatial_convergence',
            resolutions=[32, 64, 128, 256],
            dt=0.01,
            n_periods=3.0,  # Reduced from 10.0 for faster benchmarking
            M=20,
        )

    if args.mode in ['temporal', 'both']:
        temporal_convergence_study(
            output_dir=args.output_dir / 'temporal_convergence',
            resolution=64,
            n_periods=5.0,
            M=20,
        )

    print("\n" + "="*70)
    print("BENCHMARK COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {args.output_dir}")
    print("\nNext steps:")
    print("  1. Run: uv run scripts/benchmarks/analyze_alfven_dispersion.py")
    print("  2. Check paper/figures/ for publication-quality plots")
    print()


if __name__ == '__main__':
    main()
