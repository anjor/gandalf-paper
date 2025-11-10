#!/usr/bin/env python3
"""
Generate synthetic turbulent cascade data for testing analysis pipeline
Creates HDF5 files with k⊥^(-5/3) spectra plus noise
"""
import numpy as np
import h5py
from datetime import datetime
import os

def generate_kolmogorov_spectrum(k, alpha=5/3, k_min=2, k_max=50, noise_level=0.1):
    """Generate a Kolmogorov spectrum E(k) ∝ k^(-α) with noise"""

    # Base spectrum
    E_k = np.zeros_like(k, dtype=float)

    # Inertial range
    inertial = (k >= k_min) & (k <= k_max)
    E_k[inertial] = k[inertial]**(-alpha)

    # Normalize
    E_k[inertial] = E_k[inertial] / E_k[inertial].max()

    # Add forcing bump at large scales
    forcing = (k >= 2) & (k <= 5)
    E_k[forcing] *= 1.5

    # Add dissipation at small scales
    dissipation = k > k_max
    E_k[dissipation] = E_k[k_max] * np.exp(-2 * (k[dissipation] - k_max) / k_max)

    # Add noise
    noise = np.random.lognormal(0, noise_level, size=len(k))
    E_k = E_k * noise

    # Ensure positivity
    E_k = np.maximum(E_k, 1e-10)

    return E_k

def generate_cascade_data(N, n_spectra=10):
    """Generate synthetic cascade data for resolution N"""

    # Create output directory
    output_dir = f"data/benchmarks/turbulent_cascade/N{N}"
    os.makedirs(output_dir, exist_ok=True)

    # Wave numbers
    k_max = N // 3
    k_bins = np.arange(1, k_max + 1, dtype=float)

    # Time parameters
    n_periods = 50
    steady_state_wait = 20
    save_interval = 3

    # Generate time series data
    n_steps = 1000
    time = np.linspace(0, n_periods, n_steps)

    # Energy evolution (approach steady state)
    E_initial = 0.1
    E_steady = 0.05
    tau_growth = 10
    energy = E_steady + (E_initial - E_steady) * np.exp(-time / tau_growth)
    energy += 0.002 * np.sin(2 * np.pi * time / 5)  # Add oscillations

    # Helicity (small, oscillating)
    helicity = 0.01 * energy * np.sin(2 * np.pi * time / 7)

    # Dissipation and injection (balanced in steady state)
    dissipation = 0.001 * np.ones_like(time)
    injection = dissipation * (1 + 0.1 * np.sin(2 * np.pi * time / 3))

    # Generate spectra at different times
    spectra_times = np.linspace(steady_state_wait, n_periods, n_spectra)
    spectra_data = []

    for i, t in enumerate(spectra_times):
        # Generate spectrum with slightly different parameters
        alpha = 5/3 + 0.05 * np.random.randn()  # Add variation around 5/3
        noise = 0.1 + 0.05 * np.random.rand()

        E_k = generate_kolmogorov_spectrum(k_bins, alpha=alpha, k_max=N//4, noise_level=noise)
        spectra_data.append(E_k)

    # Save to HDF5
    output_file = os.path.join(output_dir, f"turbulent_cascade_N{N}_synthetic.h5")

    with h5py.File(output_file, 'w') as f:
        # Metadata
        f.attrs['N'] = N
        f.attrs['eta'] = 2.0 if N == 128 else 20.0
        f.attrs['nu'] = 0.1 if N == 128 else 1.0
        f.attrs['r'] = 2
        f.attrs['forcing_amplitude'] = 0.01
        f.attrs['n_periods'] = n_periods
        f.attrs['steady_state_wait'] = steady_state_wait
        f.attrs['timestamp'] = datetime.now().isoformat()
        f.attrs['synthetic'] = True

        # Time series
        f.create_dataset('time', data=time)
        f.create_dataset('energy', data=energy)
        f.create_dataset('helicity', data=helicity)
        f.create_dataset('dissipation', data=dissipation)
        f.create_dataset('injection', data=injection)

        # Spectra
        f.create_dataset('spectra_times', data=spectra_times)
        f.create_dataset('spectra', data=np.array(spectra_data))
        f.create_dataset('k_bins', data=k_bins)

    print(f"Generated synthetic data: {output_file}")

    # Compute and report spectral index
    avg_spectrum = np.mean(spectra_data, axis=0)
    k_fit_min = 10
    k_fit_max = min(30, N//4)

    mask = (k_bins >= k_fit_min) & (k_bins <= k_fit_max)
    if np.sum(mask) > 2:
        from scipy import stats
        log_k = np.log10(k_bins[mask])
        log_E = np.log10(avg_spectrum[mask])
        slope, _, r_value, _, std_err = stats.linregress(log_k, log_E)
        alpha = -slope
        print(f"  Spectral index: α = {alpha:.3f} ± {std_err:.3f}")
        print(f"  Expected: α = 5/3 = 1.667")

    return output_file

def main():
    """Generate synthetic data for N=64 and N=128"""
    print("Generating synthetic turbulent cascade data...")
    print("="*60)

    # Generate for N=64
    print("\nN=64 resolution:")
    file_64 = generate_cascade_data(N=64, n_spectra=8)

    # Generate for N=128
    print("\nN=128 resolution:")
    file_128 = generate_cascade_data(N=128, n_spectra=12)

    print("\n" + "="*60)
    print("Synthetic data generation complete!")
    print("\nGenerated files:")
    print(f"  - {file_64}")
    print(f"  - {file_128}")

    return [file_64, file_128]

if __name__ == "__main__":
    files = main()