"""Task 7: 2D memory and scalability analysis."""

import sys
import os
import platform
import csv
import ctypes
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"

PARTICLE_COUNTS = [100, 1000, 5000, 10000]
LARGE_PARTICLE_COUNTS = [100, 1000, 5000, 10000, 50000, 100000]
DIMENSIONS = 2

DATA_TYPES = {
    "float32": np.float32,
    "float64": np.float64,
}


# Calculate theoretical pairwise displacement memory
def calculate_displacement_memory(N: int, dtype) -> int:
    return int(N * N * DIMENSIONS * np.dtype(dtype).itemsize)


calculate_memory = calculate_displacement_memory


# Format bytes to MiB or GiB string
def format_memory(size_bytes: int) -> str:
    mib = size_bytes / (1024 ** 2)
    gib = size_bytes / (1024 ** 3)
    if gib >= 1.0:
        return f"{gib:.3f} GiB"
    return f"{mib:.3f} MiB"


# Print position array strides and memory layout
def print_memory_layout():
    np.random.seed(0)
    N_test = 12
    pos64 = np.random.rand(N_test, DIMENSIONS) * 10.0
    pos32 = pos64.astype(np.float32)

    print("\n1. MEMORY LAYOUT OF POSITION ARRAY (2D)")
    print("-" * 55)
    print("Shape       :", pos64.shape)
    print("Data type   :", pos64.dtype)
    print("Item size   :", pos64.itemsize, "bytes")
    print("Total bytes :", pos64.nbytes, "bytes")
    print("Strides     :", pos64.strides)

    print("\nFLOAT32 POSITION ARRAY")
    print("Data type   :", pos32.dtype)
    print("Item size   :", pos32.itemsize, "bytes")
    print("Total bytes :", pos32.nbytes, "bytes")
    print("Strides     :", pos32.strides)

    c_float_sz = ctypes.sizeof(ctypes.c_float)
    c_double_sz = ctypes.sizeof(ctypes.c_double)
    print("\nC DATA TYPE SIZES")
    print("sizeof(float) :", c_float_sz, "bytes")
    print("sizeof(double):", c_double_sz, "bytes")
    print("Match NumPy float32 == C float :", pos32.itemsize == c_float_sz)
    print("Match NumPy float64 == C double:", pos64.itemsize == c_double_sz)


# Print large position array memory for N=100k
def print_large_position_memory():
    N_large = 100000
    c_float_sz = ctypes.sizeof(ctypes.c_float)
    c_double_sz = ctypes.sizeof(ctypes.c_double)

    mem64_bytes = N_large * DIMENSIONS * c_double_sz
    mem32_bytes = N_large * DIMENSIONS * c_float_sz

    print(f"\n2. POSITION ARRAY MEMORY FOR N = {N_large:,} PARTICLES (2D)")
    print("-" * 55)
    print(f"Float64: {mem64_bytes:,} bytes ({mem64_bytes / (1024**2):.3f} MiB)")
    print(f"Float32: {mem32_bytes:,} bytes ({mem32_bytes / (1024**2):.3f} MiB)")


# Print pairwise displacement memory table
def print_pairwise_memory_table():
    print("\n3. PAIRWISE DISPLACEMENT MEMORY TABLE (2D)")
    print(f"Array shape: (N, N, {DIMENSIONS}) | Formula: N x N x {DIMENSIONS} x itemsize")
    print("-" * 55)
    print(f"{'N':>8} | {'2D Float32':>16} | {'2D Float64':>16}")
    print("-" * 55)
    for N in PARTICLE_COUNTS:
        mem32 = calculate_displacement_memory(N, np.float32)
        mem64 = calculate_displacement_memory(N, np.float64)
        print(f"{N:8d} | {format_memory(mem32):>16} | {format_memory(mem64):>16}")


# Verify theoretical byte count against NumPy nbytes
def verify_numpy_memory(N_test: int = 1000):
    print(f"\n4. NUMPY ALLOCATION VERIFICATION (N = {N_test:,}, 2D)")
    print("-" * 65)
    print(f"{'Dtype':<10} | {'Theoretical (bytes)':<22} | {'NumPy nbytes':<16} | {'Match':<6}")
    print("-" * 65)
    for dtype_name, dtype in DATA_TYPES.items():
        arr = np.zeros((N_test, N_test, DIMENSIONS), dtype=dtype)
        theoretical = calculate_displacement_memory(N_test, dtype)
        actual = int(arr.nbytes)
        match = (theoretical == actual)
        print(f"{dtype_name:<10} | {theoretical:<22,d} | {actual:<16,d} | {str(match):<6}")
        del arr


# Print out-of-memory analysis
def print_oom_analysis(N_oom: int = 50000):
    mem32_bytes = calculate_displacement_memory(N_oom, np.float32)
    mem64_bytes = calculate_displacement_memory(N_oom, np.float64)

    print(f"\n5. OUT-OF-MEMORY (OOM) LIMIT ANALYSIS (N = {N_oom:,}, 2D)")
    print("-" * 65)
    print(f"At N = {N_oom:,}, a single dense (N, N, 2) displacement tensor requires:")
    print(f"  Float32: {mem32_bytes / (1024**3):6.2f} GiB ({mem32_bytes / 1e9:5.1f} GB decimal)")
    print(f"  Float64: {mem64_bytes / (1024**3):6.2f} GiB ({mem64_bytes / 1e9:5.1f} GB decimal)")


# Save environment specs
def save_environment_info(results_dir: Path):
    env_file = results_dir / "task7_environment.txt"
    with open(env_file, "w") as f:
        f.write("TASK 7: ENVIRONMENT AND SYSTEM SPECIFICATIONS\n")
        f.write("=" * 55 + "\n")
        f.write(f"Date & Time        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Operating System   : {platform.system()} {platform.release()} ({platform.machine()})\n")
        f.write(f"Python Version     : {platform.python_version()}\n")
        f.write(f"NumPy Version      : {np.__version__}\n")
        f.write(f"Matplotlib Version : {matplotlib.__version__}\n")
        f.write(f"Simulation Space   : 2D (x, y)\n")


# Export CSV tables
def export_csv_data(results_dir: Path):
    meas_csv = results_dir / "memory_measurements.csv"
    with open(meas_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["N_particles", "dimensions", "dtype", "itemsize_bytes", "total_bytes", "memory_mib", "memory_formatted"])
        for N in LARGE_PARTICLE_COUNTS:
            for dtype_name, dtype in DATA_TYPES.items():
                itemsize = np.dtype(dtype).itemsize
                tot_bytes = N * DIMENSIONS * itemsize
                mib = tot_bytes / (1024 ** 2)
                writer.writerow([N, DIMENSIONS, dtype_name, itemsize, tot_bytes, f"{mib:.4f}", format_memory(tot_bytes)])

    est_csv = results_dir / "pairwise_memory_estimates.csv"
    with open(est_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["N_particles", "array_shape", "dtype", "formula", "total_bytes", "memory_mib", "memory_gib", "memory_formatted"])
        for N in LARGE_PARTICLE_COUNTS:
            for dtype_name, dtype in DATA_TYPES.items():
                tot_bytes = calculate_displacement_memory(N, dtype)
                mib = tot_bytes / (1024 ** 2)
                gib = tot_bytes / (1024 ** 3)
                writer.writerow([N, f"({N}, {N}, {DIMENSIONS})", dtype_name, f"{N}x{N}x{DIMENSIONS}x{np.dtype(dtype).itemsize}", tot_bytes, f"{mib:.4f}", f"{gib:.4f}", format_memory(tot_bytes)])


# Generate 2D memory scaling plots
def generate_plots(results_dir: Path):
    results_dir.mkdir(parents=True, exist_ok=True)
    N_values = np.array(PARTICLE_COUNTS)

    # Plot 1: Position memory comparison
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bar_width = 0.35
    x_indices = np.arange(len(PARTICLE_COUNTS))
    pos32_kb = [(N * DIMENSIONS * 4) / 1024 for N in PARTICLE_COUNTS]
    pos64_kb = [(N * DIMENSIONS * 8) / 1024 for N in PARTICLE_COUNTS]

    ax.bar(x_indices - bar_width / 2, pos32_kb, bar_width, label="Float32", color="#1f77b4")
    ax.bar(x_indices + bar_width / 2, pos64_kb, bar_width, label="Float64", color="#ff7f0e")
    ax.set_title("Position Array Memory (2D)")
    ax.set_xlabel("Particle Count (N)")
    ax.set_ylabel("Memory (KiB)")
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f"N={N:,}" for N in PARTICLE_COUNTS])
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "position_memory_comparison.png", dpi=200)
    plt.close(fig)

    # Plot 2: Pairwise displacement scalability
    N_continuous = np.logspace(2, 4.8, 150)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for dtype_name, color in [("float32", "#1f77b4"), ("float64", "#ff7f0e")]:
        dtype = DATA_TYPES[dtype_name]
        mem_curve_gib = (N_continuous ** 2 * DIMENSIONS * np.dtype(dtype).itemsize) / (1024 ** 3)
        ax.loglog(N_continuous, mem_curve_gib, label=f"{dtype_name} (2D)", color=color, lw=2.2)

    ax.axhline(8.0, color="orange", linestyle="--", lw=1.2, label="8 GB RAM")
    ax.axhline(16.0, color="crimson", linestyle="--", lw=1.4, label="16 GB RAM")
    ax.axhline(32.0, color="darkred", linestyle=":", lw=1.4, label="32 GB RAM")
    ax.set_title("Pairwise Displacement Memory Scalability (2D)")
    ax.set_xlabel("Number of Particles (N)")
    ax.set_ylabel("Memory (GiB)")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(results_dir / "pairwise_memory_scalability.png", dpi=200)
    fig.savefig(results_dir / "memory_and_scalability_fsc.png", dpi=200)
    plt.close(fig)

    # Plot 3: Growth factors
    growth_factors = (N_values[1:] / N_values[:-1]) ** 2
    interval_labels = [f"{N_values[i]:,} to {N_values[i+1]:,}" for i in range(len(N_values) - 1)]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.bar(interval_labels, growth_factors, width=0.55, color="#2ca02c")
    ax.set_title("Memory Multiplication Factors (2D)")
    ax.set_xlabel("Particle Count Transition")
    ax.set_ylabel("Multiplication Factor")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, value in zip(bars, growth_factors):
        ax.annotate(
            f"{value:.0f}x",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            fontweight="bold"
        )
    ax.set_ylim(0, max(growth_factors) * 1.2)
    fig.tight_layout()
    fig.savefig(results_dir / "memory_growth_fsc.png", dpi=200)
    plt.close(fig)


# Main runner
def main():
    print("Task 7: Memory and Scalability Analysis (2D)")
    print("-" * 55)

    print_memory_layout()
    print_large_position_memory()
    print_pairwise_memory_table()
    verify_numpy_memory(N_test=1000)
    print_oom_analysis(N_oom=50000)

    DEFAULT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_environment_info(DEFAULT_RESULTS_DIR)
    export_csv_data(DEFAULT_RESULTS_DIR)
    generate_plots(DEFAULT_RESULTS_DIR)

    # Mirror to task7_results for backward compatibility
    legacy_dir = SCRIPT_DIR / "task7_results"
    if DEFAULT_RESULTS_DIR != legacy_dir:
        legacy_dir.mkdir(parents=True, exist_ok=True)
        for f in DEFAULT_RESULTS_DIR.glob("*"):
            if f.is_file():
                import shutil
                shutil.copy2(f, legacy_dir / f.name)

    print(f"\nResults saved in {DEFAULT_RESULTS_DIR}")
    print("-" * 55)


run_task_7 = main

if __name__ == "__main__":
    main()
