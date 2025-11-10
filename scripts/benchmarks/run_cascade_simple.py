#!/usr/bin/env python3
"""
Simple turbulent cascade simulation for k⊥^(-5/3) spectrum verification
Based directly on GANDALF's driven_turbulence.py example
"""
import numpy as np
import jax
import jax.numpy as jnp
import h5py
import time
from datetime import datetime
import sys
import os
from pathlib import Path

# Add GANDALF to path
gandalf_path = os.environ.get('GANDALF_PATH')
if not gandalf_path:
    # Try default location
    default_path = os.path.expanduser("~/repos/anjor/gandalf/src")
    if os.path.exists(default_path):
        gandalf_path = default_path
    else:
        raise RuntimeError(
            "GANDALF_PATH environment variable not set. "
            "Please set it to the src/ directory of your GANDALF installation:\n"
            "export GANDALF_PATH=/path/to/gandalf/src"
        )
sys.path.insert(0, gandalf_path)

from krmhd import (
    SpectralGrid3D,
    initialize_random_spectrum,
    gandalf_step,
    compute_cfl_timestep,
    energy_spectrum_perpendicular,
)
from krmhd.diagnostics import compute_energy, EnergyHistory
from krmhd.forcing import force_alfven_modes, compute_energy_injection_rate


def run_cascade_simulation():
    """Run turbulent cascade simulation"""

    # Resolution parameters
    N = 64  # Start with N=64 for quick test

    # Physical parameters (from GANDALF examples)
    v_A = 1.0           # Alfvén speed
    beta_i = 1.0        # Ion pressure ratio
    eta = 20.0 if N == 64 else 2.0  # N=64 anomaly from Issue #82
    nu = 1.0 if N == 64 else 0.1    # Viscosity

    # Grid parameters
    Nx = Ny = Nz = N
    Lx = Ly = Lz = 2 * np.pi

    # Initial spectrum parameters
    alpha_init = 2.0    # Initial spectral index
    amplitude = 1e-3    # Initial amplitude (weak)
    k_min = 1
    k_max = N // 3

    # Forcing parameters (inject at large scales)
    force_amplitude = 1e-4  # Gentle forcing
    k_force_min = 2
    k_force_max = 5

    # Time stepping
    cfl_safety = 0.3
    n_periods = 50      # Run for 50 Alfvén times
    steady_wait = 20    # Wait 20 τ_A for steady state

    print(f"\n{'='*70}")
    print(f"Turbulent Cascade Simulation")
    print(f"{'='*70}")
    print(f"Resolution: {N}³")
    print(f"Dissipation: η={eta}, ν={nu}")
    print(f"Forcing: k ∈ [{k_force_min}, {k_force_max}], amplitude={force_amplitude}")
    print(f"Total time: {n_periods} τ_A")
    print(f"{'='*70}\n")

    # Create output directory
    output_dir = Path("data/benchmarks/turbulent_cascade")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize grid
    grid = SpectralGrid3D.create(Nx=Nx, Ny=Ny, Nz=Nz, Lx=Lx, Ly=Ly, Lz=Lz)
    print(f"✓ Created {Nx}×{Ny}×{Nz} spectral grid")

    # Initialize state with random spectrum
    state = initialize_random_spectrum(
        grid,
        M=20,              # Number of Hermite moments
        alpha=alpha_init,
        amplitude=amplitude,
        k_min=k_min,
        k_max=k_max,
        v_th=1.0,
        beta_i=beta_i,
        nu=nu,
        Lambda=1.0,
        seed=42,
    )
    print(f"✓ Initialized weak k^(-{alpha_init:.2f}) spectrum")

    # Initialize JAX random key for forcing
    key = jax.random.PRNGKey(42)

    # Compute initial energy
    initial_energies = compute_energy(state)
    print(f"  Initial energy: E_total = {initial_energies['total']:.6e}")

    # Initialize energy history
    history = EnergyHistory()
    history.append(state)

    # Storage for spectra after steady state
    spectra_times = []
    spectra_data = []

    # Time parameters
    t_alfven = 2 * np.pi / v_A
    t_final = n_periods * t_alfven
    t_steady = steady_wait * t_alfven
    save_interval = 5 * t_alfven  # Save spectrum every 5 τ_A after steady state

    # Compute CFL-limited timestep
    dt = compute_cfl_timestep(state, v_A=v_A, cfl_safety=cfl_safety)
    print(f"  Using dt = {dt:.4f} (CFL-limited)")

    # Track energy injection
    total_injection = 0.0
    t = 0.0
    step = 0
    t_next_save = t_steady + save_interval

    start_wall_time = time.time()

    print(f"\nRunning simulation...")

    # Main timestepping loop
    while t < t_final:
        # Apply forcing (add energy at large scales)
        state_before_forcing = state
        state, key = force_alfven_modes(
            state,
            amplitude=force_amplitude,
            k_min=k_force_min,
            k_max=k_force_max,
            dt=dt,
            key=key
        )

        # Compute energy injection rate
        eps_inj = compute_energy_injection_rate(state_before_forcing, state, dt)
        total_injection += eps_inj * dt

        # Evolve dynamics (cascade + dissipation)
        state = gandalf_step(state, dt=dt, eta=eta, v_A=v_A)

        t += dt
        step += 1

        # Save spectrum if in steady state
        if t >= t_steady and t >= t_next_save:
            spec_perp = energy_spectrum_perpendicular(state)
            spectra_times.append(t / t_alfven)
            spectra_data.append({
                'k_perp': np.array(spec_perp['k_perp']),
                'E_perp': np.array(spec_perp['E_perp'])
            })
            t_next_save = t + save_interval
            print(f"  Saved spectrum at t={t/t_alfven:.1f} τ_A")

        # Progress update
        if step % 100 == 0:
            E = compute_energy(state)["total"]
            percent = 100 * t / t_final
            elapsed = time.time() - start_wall_time
            eta_seconds = elapsed * (t_final - t) / t if t > 0 else 0
            print(f"  Step {step:4d}: t={t/t_alfven:.2f} τ_A ({percent:.1f}%), "
                  f"E={E:.4e}, ε_inj={eps_inj:.3e}, ETA: {eta_seconds/60:.1f} min")

            # Check for NaN
            if np.isnan(E):
                print("ERROR: Energy is NaN, simulation diverged!")
                break

        # Save to history periodically
        if step % 50 == 0:
            history.append(state)

    elapsed_time = time.time() - start_wall_time

    print(f"\n✓ Completed simulation")
    print(f"  Runtime: {elapsed_time:.1f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"  Final time: {t/t_alfven:.1f} τ_A")
    print(f"  Number of spectra saved: {len(spectra_data)}")

    # Compute final diagnostics
    final_energies = compute_energy(state)
    print(f"\nFinal diagnostics:")
    print(f"  Final energy: E_total = {final_energies['total']:.6e}")
    print(f"  Energy change: ΔE = {final_energies['total'] - initial_energies['total']:.6e}")
    print(f"  Total injection: ∫ε_inj dt = {total_injection:.6e}")

    # Save results
    output_file = output_dir / f"cascade_N{N}_simple.h5"

    with h5py.File(output_file, 'w') as f:
        # Metadata
        f.attrs['N'] = N
        f.attrs['eta'] = eta
        f.attrs['nu'] = nu
        f.attrs['forcing_amplitude'] = force_amplitude
        f.attrs['n_periods'] = n_periods
        f.attrs['steady_wait'] = steady_wait
        f.attrs['timestamp'] = datetime.now().isoformat()

        # Time series from history
        f.create_dataset('history_times', data=np.array(history.times))
        f.create_dataset('history_E_total', data=np.array(history.E_total))
        f.create_dataset('history_E_magnetic', data=np.array(history.E_magnetic))
        f.create_dataset('history_E_kinetic', data=np.array(history.E_kinetic))

        # Spectra data
        f.create_dataset('spectra_times', data=np.array(spectra_times))

        if len(spectra_data) > 0:
            # Save all spectra
            k_perp = spectra_data[0]['k_perp']
            E_perp_all = np.array([s['E_perp'] for s in spectra_data])
            f.create_dataset('k_perp', data=k_perp)
            f.create_dataset('E_perp_all', data=E_perp_all)

            # Compute and save average spectrum
            E_perp_avg = np.mean(E_perp_all, axis=0)
            f.create_dataset('E_perp_avg', data=E_perp_avg)

            # Fit power law
            from scipy import stats
            k_fit_min = 10
            k_fit_max = 30

            mask = (k_perp >= k_fit_min) & (k_perp <= k_fit_max)
            if np.sum(mask) > 2:
                log_k = np.log10(k_perp[mask])
                log_E = np.log10(E_perp_avg[mask] + 1e-20)
                slope, intercept, r_value, p_value, std_err = stats.linregress(log_k, log_E)
                alpha_fit = -slope

                f.attrs['spectral_index'] = alpha_fit
                f.attrs['spectral_index_err'] = std_err
                f.attrs['fit_r_value'] = r_value

                print(f"\nSpectral fit in k ∈ [{k_fit_min}, {k_fit_max}]:")
                print(f"  α = {alpha_fit:.3f} ± {std_err:.3f}")
                print(f"  Expected: α = 5/3 = 1.667")
                print(f"  Deviation: {abs(alpha_fit - 5/3):.3f}")

    print(f"\n✓ Results saved to: {output_file}")
    return output_file


if __name__ == "__main__":
    output_file = run_cascade_simulation()
    print(f"\n{'='*70}")
    print(f"Simulation complete!")
    print(f"{'='*70}\n")