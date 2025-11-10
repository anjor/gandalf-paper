#!/usr/bin/env -S uv run python
"""
Orszag-Tang Vortex Benchmark for GANDALF Paper

Runs spatial and temporal convergence studies for nonlinear Orszag-Tang vortex.
Tests nonlinear MHD dynamics, energy conservation, and cascade physics.

This script generates benchmark data for the Verification section of the paper.

Usage:
    uv run scripts/benchmarks/run_orszag_tang.py --mode spatial
    uv run scripts/benchmarks/run_orszag_tang.py --mode temporal
    uv run scripts/benchmarks/run_orszag_tang.py --mode both

Output:
    data/benchmarks/orszag_tang/spatial_convergence/*.h5
    data/benchmarks/orszag_tang/temporal_convergence/*.h5

Physics:
    The Orszag-Tang vortex is a standard benchmark for nonlinear MHD dynamics.
    Initial conditions (incompressible RMHD formulation):

    Stream function: φ = -2[cos(kx·x) + cos(ky·y)]
    Vector potential: Ψ = B0·[cos(2kx·x) + 2cos(ky·y)]

    where kx = 2π/Lx, ky = 2π/Ly and B0 = 1/√(4π).

    Expected dynamics:
    - Nonlinear energy cascade from large to small scales
    - Formation of current sheets (analog of shocks)
    - Selective decay: magnetic energy increases relative to kinetic
    - Energy conservation: |ΔE/E₀| < 0.01% for inviscid case

    Convergence rates:
    - Spatial (spectral): Exponential convergence E ~ exp(-αN)
    - Temporal (RK2): Second-order convergence E ~ O(Δt²)

References:
    - Orszag & Tang (1979) J. Fluid Mech. 90:129 - Original paper
    - Schekochihin et al. (2009) ApJS 182:310 - KRMHD formulation
"""

import sys
import os
from pathlib import Path
import argparse
import numpy as np
import h5py
from typing import Dict, List
import time

# Add GANDALF to path
# Priority 1: Check GANDALF_PATH environment variable
# Priority 2: Try relative path (development setup)
gandalf_path = None
if "GANDALF_PATH" in os.environ:
    gandalf_path = Path(os.environ["GANDALF_PATH"]) / "src"
    if not gandalf_path.exists():
        raise FileNotFoundError(
            f"GANDALF_PATH environment variable set to {os.environ['GANDALF_PATH']}, "
            f"but source not found at {gandalf_path}"
        )
else:
    # Development setup: gandalf-paper/scripts/benchmarks/ and gandalf/ as siblings
    gandalf_path = Path(__file__).resolve().parents[3] / "gandalf" / "src"
    if not gandalf_path.exists():
        raise FileNotFoundError(
            f"GANDALF source not found at {gandalf_path}\n"
            f"Expected directory structure:\n"
            f"  parent/\n"
            f"    gandalf/src/  (GANDALF source code)\n"
            f"    gandalf-paper/scripts/benchmarks/  (this script)\n"
            f"Alternatively, set GANDALF_PATH environment variable:\n"
            f"  export GANDALF_PATH=/path/to/gandalf\n"
            f"  uv run scripts/benchmarks/run_orszag_tang.py"
        )

if str(gandalf_path) not in sys.path:
    sys.path.insert(0, str(gandalf_path))

import jax
import jax.numpy as jnp

from krmhd.physics import initialize_orszag_tang, KRMHDState, energy
from krmhd.spectral import SpectralGrid3D
from krmhd.timestepping import gandalf_step, compute_cfl_timestep


def run_orszag_tang(
    grid: SpectralGrid3D,
    M: int,
    resolution: int,
    dt: float,
    n_periods: float = 3.0,
    B0: float = None,
    v_A: float = 1.0,
    eta: float = 0.0,
    nu: float = 0.0,
    cfl_safety: float = 0.3,
) -> Dict:
    """
    Run single Orszag-Tang vortex simulation.

    Args:
        grid: Spectral grid
        M: Number of Hermite moments (minimum 2 for GANDALF collision operator)
        resolution: Grid resolution (N for N² × 2 grid)
        dt: Time step (if None, use CFL-limited adaptive stepping)
        n_periods: Number of Alfvén crossing times to integrate
        B0: Magnetic field amplitude (default: 1/√(4π))
        v_A: Alfvén velocity
        eta: Resistivity
        nu: Collision frequency
        cfl_safety: CFL safety factor for adaptive timestepping

    Returns:
        Dictionary with:
            - times: Time array
            - energy_total: Total energy time series
            - energy_kinetic: Kinetic energy time series
            - energy_magnetic: Magnetic energy time series
            - energy_conservation_error: |ΔE/E₀|
            - z_plus_snapshots: List of z+ field snapshots
            - z_minus_snapshots: List of z- field snapshots
            - snapshot_times: Times of snapshots
            - resolution: Grid resolution
            - dt_initial: Initial timestep
            - n_steps: Number of timesteps taken
    """
    print(f"  Running N={resolution}² × 2, dt={dt:.6f}...")

    # Initialize Orszag-Tang vortex
    # M=2 (minimum): Pure fluid RMHD with nu=0 (no kinetic effects, no collisions)
    # eta=0: Inviscid for energy conservation test
    if B0 is None:
        B0 = 1.0 / np.sqrt(4.0 * np.pi)

    state = initialize_orszag_tang(
        grid=grid,
        M=M,
        B0=B0,
        v_th=1.0,
        beta_i=1.0,
        nu=nu,
        Lambda=1.0,
    )

    # Time normalization: τ_A = Lz/v_A (Alfvén crossing time)
    # With Lz = 1.0, v_A = 1.0: τ_A = 1.0 time units
    tau_A = grid.Lz / v_A
    t_final = n_periods * tau_A

    # Storage for diagnostics
    times = []
    energies_total = []
    energies_kinetic = []
    energies_magnetic = []

    # Storage for field snapshots (save at t=0, τ_A, 2τ_A, 3τ_A)
    snapshot_times = []
    z_plus_snapshots = []
    z_minus_snapshots = []
    snapshot_interval = tau_A  # Save every Alfvén time
    next_snapshot_time = 0.0

    # Initial energy
    E0_dict = energy(state)
    E0 = E0_dict['total']

    print(f"    Initial energy: E_total = {E0:.6e}")
    print(f"    E_mag/E_kin = {E0_dict['magnetic']/E0_dict['kinetic']:.3f}")

    # Time integration loop
    step = 0
    max_steps = 100000  # Safety limit

    # Use fixed dt if provided, else adaptive CFL
    use_adaptive_dt = (dt is None)
    if use_adaptive_dt:
        dt = compute_cfl_timestep(state, v_A, cfl_safety)
        dt = min(dt, t_final - state.time)
        print(f"    Using adaptive timestepping with CFL safety = {cfl_safety}")
    else:
        print(f"    Using fixed timestep dt = {dt:.6f}")

    dt_initial = dt

    while state.time < t_final and step < max_steps:
        # Record diagnostics
        times.append(float(state.time))

        E_dict = energy(state)
        energies_total.append(float(E_dict['total']))
        energies_kinetic.append(float(E_dict['kinetic']))
        energies_magnetic.append(float(E_dict['magnetic']))

        # Save field snapshots at specified intervals
        if state.time >= next_snapshot_time:
            snapshot_times.append(float(state.time))
            # Save z± fields at kz=0 plane (2D problem)
            z_plus_snapshots.append(np.array(state.z_plus[0, :, :]))
            z_minus_snapshots.append(np.array(state.z_minus[0, :, :]))
            next_snapshot_time += snapshot_interval

        # Adaptive timestepping
        if use_adaptive_dt:
            dt = compute_cfl_timestep(state, v_A, cfl_safety)

        # Don't overshoot final time
        dt = min(dt, t_final - state.time)

        # Evolve
        state = gandalf_step(state, dt=dt, eta=eta, v_A=v_A)
        step += 1

    # Final snapshot
    if len(snapshot_times) == 0 or snapshot_times[-1] < state.time:
        snapshot_times.append(float(state.time))
        z_plus_snapshots.append(np.array(state.z_plus[0, :, :]))
        z_minus_snapshots.append(np.array(state.z_minus[0, :, :]))

    # Convert to arrays
    times = np.array(times)
    energies_total = np.array(energies_total)
    energies_kinetic = np.array(energies_kinetic)
    energies_magnetic = np.array(energies_magnetic)

    # Compute energy conservation error
    E_final = energies_total[-1]
    energy_error = np.abs(E_final - E0) / E0

    print(f"    Final time: t = {state.time:.3f} ({state.time/tau_A:.2f} τ_A)")
    print(f"    Steps: {step}")
    print(f"    Energy conservation: |ΔE/E₀| = {energy_error:.2e}")
    print(f"    E_mag/E_kin (final) = {energies_magnetic[-1]/energies_kinetic[-1]:.3f}")

    return {
        'times': times,
        'energy_total': energies_total,
        'energy_kinetic': energies_kinetic,
        'energy_magnetic': energies_magnetic,
        'energy_conservation_error': energy_error,
        'z_plus_snapshots': z_plus_snapshots,
        'z_minus_snapshots': z_minus_snapshots,
        'snapshot_times': snapshot_times,
        'resolution': resolution,
        'dt_initial': dt_initial,
        'n_steps': step,
        'E0': E0,
    }


def spatial_convergence_study(
    output_dir: Path,
    resolutions: List[int] = [32, 64, 128],
    dt: float = 0.01,
    n_periods: float = 3.0,
    M: int = 2,
) -> None:
    """
    Run spatial convergence study: vary resolution, fixed timestep.

    Tests spectral method convergence for nonlinear dynamics.

    Args:
        output_dir: Directory to save results
        resolutions: List of grid resolutions (N for N² × 2 grid)
        dt: Fixed timestep (well-resolved)
        n_periods: Number of Alfvén periods to integrate
        M: Number of Hermite moments (minimum 2; pure fluid with nu=0)
    """
    print("\n" + "="*70)
    print("SPATIAL CONVERGENCE STUDY")
    print("="*70)
    print(f"Resolutions: {resolutions}")
    print(f"Fixed dt = {dt}")
    print(f"Integration time: {n_periods} Alfvén periods")
    print(f"M = {M} (pure fluid RMHD, nu=0)")
    print()

    output_dir.mkdir(parents=True, exist_ok=True)

    for N in resolutions:
        start_time = time.time()

        # Create grid (2D problem: N² × 2)
        Lx = Ly = Lz = 1.0  # Unit box
        grid = SpectralGrid3D.create(Nx=N, Ny=N, Nz=2, Lx=Lx, Ly=Ly, Lz=Lz)

        # Run simulation
        results = run_orszag_tang(
            grid=grid,
            M=M,
            resolution=N,
            dt=dt,
            n_periods=n_periods,
            v_A=1.0,
            eta=0.0,  # Inviscid
            nu=0.0,
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
            f.attrs['E0'] = results['E0']
            f.attrs['n_steps'] = results['n_steps']

            # Convergence metrics
            f.create_dataset('energy_conservation_error', data=results['energy_conservation_error'])

            # Time series
            f.create_dataset('times', data=results['times'], compression='gzip')
            f.create_dataset('energy_total', data=results['energy_total'], compression='gzip')
            f.create_dataset('energy_kinetic', data=results['energy_kinetic'], compression='gzip')
            f.create_dataset('energy_magnetic', data=results['energy_magnetic'], compression='gzip')

            # Field snapshots
            f.create_dataset('snapshot_times', data=results['snapshot_times'])

            # Save snapshots as 3D arrays (n_snapshots, Ny, Nx_rfft)
            z_plus_array = np.array(results['z_plus_snapshots'])
            z_minus_array = np.array(results['z_minus_snapshots'])
            f.create_dataset('z_plus_snapshots', data=z_plus_array, compression='gzip')
            f.create_dataset('z_minus_snapshots', data=z_minus_array, compression='gzip')

        elapsed = time.time() - start_time
        print(f"  Saved to {output_file} ({elapsed:.1f}s)\n")

    print("Spatial convergence study complete!")


def temporal_convergence_study(
    output_dir: Path,
    timesteps: List[float] = None,
    resolution: int = 64,
    n_periods: float = 2.0,
    M: int = 2,
) -> None:
    """
    Run temporal convergence study: vary timestep, fixed resolution.

    Tests GANDALF integrating factor + RK2 convergence for nonlinear dynamics.

    Args:
        output_dir: Directory to save results
        timesteps: List of timesteps to test
        resolution: Fixed grid resolution (N for N² × 2)
        n_periods: Number of Alfvén periods to integrate
        M: Number of Hermite moments (minimum 2; pure fluid with nu=0)
    """
    print("\n" + "="*70)
    print("TEMPORAL CONVERGENCE STUDY")
    print("="*70)
    print(f"Fixed resolution: {resolution}² × 2")
    print(f"Integration time: {n_periods} Alfvén periods")
    print(f"M = {M} (pure fluid RMHD, nu=0)")
    print()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create grid once (fixed resolution)
    Lx = Ly = Lz = 1.0
    grid = SpectralGrid3D.create(Nx=resolution, Ny=resolution, Nz=2, Lx=Lx, Ly=Ly, Lz=Lz)

    # Determine timesteps if not provided
    if timesteps is None:
        # Alfvén crossing time: τ_A = Lz/v_A = 1.0
        tau_A = 1.0
        # Test temporal convergence with moderate timesteps
        timesteps = [0.05, 0.025, 0.0125]

    print(f"Timesteps: {[f'{dt:.6f}' for dt in timesteps]}")
    print()

    for dt in timesteps:
        start_time = time.time()

        # Run simulation
        results = run_orszag_tang(
            grid=grid,
            M=M,
            resolution=resolution,
            dt=dt,
            n_periods=n_periods,
            v_A=1.0,
            eta=0.0,  # Inviscid
            nu=0.0,
        )

        # Save to HDF5
        dt_str = f"{dt:.6e}".replace('.', 'p').replace('+', '').replace('-', 'm')
        output_file = output_dir / f"dt_{dt_str}.h5"

        with h5py.File(output_file, 'w') as f:
            # Metadata
            f.attrs['resolution'] = resolution
            f.attrs['dt'] = dt
            f.attrs['n_periods'] = n_periods
            f.attrs['M'] = M
            f.attrs['study_type'] = 'temporal_convergence'
            f.attrs['E0'] = results['E0']
            f.attrs['n_steps'] = results['n_steps']

            # Convergence metrics
            f.create_dataset('energy_conservation_error', data=results['energy_conservation_error'])

            # Time series
            f.create_dataset('times', data=results['times'], compression='gzip')
            f.create_dataset('energy_total', data=results['energy_total'], compression='gzip')
            f.create_dataset('energy_kinetic', data=results['energy_kinetic'], compression='gzip')
            f.create_dataset('energy_magnetic', data=results['energy_magnetic'], compression='gzip')

            # Field snapshots
            f.create_dataset('snapshot_times', data=results['snapshot_times'])

            # Save snapshots
            z_plus_array = np.array(results['z_plus_snapshots'])
            z_minus_array = np.array(results['z_minus_snapshots'])
            f.create_dataset('z_plus_snapshots', data=z_plus_array, compression='gzip')
            f.create_dataset('z_minus_snapshots', data=z_minus_array, compression='gzip')

        elapsed = time.time() - start_time
        print(f"  Saved to {output_file} ({elapsed:.1f}s)\n")

    print("Temporal convergence study complete!")


def main():
    """Main entry point for benchmark script."""
    parser = argparse.ArgumentParser(
        description='Run Orszag-Tang vortex benchmark for GANDALF paper',
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
        default=Path(__file__).resolve().parents[2] / 'data' / 'benchmarks' / 'orszag_tang',
        help='Output directory for benchmark data'
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print("ORSZAG-TANG VORTEX BENCHMARK")
    print("="*70)
    print(f"Output directory: {args.output_dir}")
    print(f"JAX devices: {jax.devices()}")
    print("="*70)

    # Run requested studies
    if args.mode in ['spatial', 'both']:
        spatial_convergence_study(
            output_dir=args.output_dir / 'spatial_convergence',
            resolutions=[32, 64, 128],
            dt=0.01,  # Well-resolved fixed timestep
            n_periods=3.0,  # Three Alfvén crossing times
            M=2,  # Pure fluid RMHD (minimum M=2, nu=0 for no collisions)
        )

    if args.mode in ['temporal', 'both']:
        temporal_convergence_study(
            output_dir=args.output_dir / 'temporal_convergence',
            timesteps=[0.05, 0.025, 0.0125],  # Test O(dt²) convergence
            resolution=64,
            n_periods=2.0,  # Two Alfvén crossing times
            M=2,  # Pure fluid RMHD (minimum M=2, nu=0 for no collisions)
        )

    print("\n" + "="*70)
    print("BENCHMARK COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {args.output_dir}")
    print("\nNext steps:")
    print("  1. Run: uv run scripts/benchmarks/analyze_orszag_tang.py")
    print("  2. Check paper/figures/ for publication-quality plots")
    print()


if __name__ == '__main__':
    main()
