"""
Task 7: Memory and Scalability Analysis (2D and 3D)
===================================================
Analyzes the theoretical and practical memory footprints of pairwise displacement
tensors in 2D (2 * N^2 * itemsize) and 3D (3 * N^2 * itemsize), verifies scaling
against NumPy array.nbytes, and evaluates the Out-Of-Memory (OOM) limit (N ~ 50,000).

Outputs (saved in task7_results/):
- task7_memory_scalability_3d.png
- task7_precision_comparison_3d.png
- task7_memory_scalability.png
- task7_precision_comparison.png
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "task7_results"

DEFAULT_N_VALUES = [100, 1000, 5000, 10000]
DATA_TYPES = {
    "float32": np.float32,
    "float64": np.float64,
}


def calculate_memory(N: int, dtype, dim: int = 3) -> int:
    """Computes theoretical memory in bytes for pairwise displacement tensor: dim * N^2 * itemsize."""
    return int(N * N * dim * np.dtype(dtype).itemsize)


def run_task_7(save_dir=None, show_plots: bool = False):
    """
    Executes Task 7 Memory Scalability & Precision Analysis for both 2D and 3D.
    Safe for automated invocations (non-blocking).
    """
    if save_dir is None:
        save_dir = DEFAULT_RESULTS_DIR
    else:
        save_dir = Path(save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("TASK 7: 2D & 3D MEMORY AND SCALABILITY ANALYSIS")
    print("=" * 78)

    # 1. Theoretical Memory Calculation across 2D and 3D
    results_mb = {}
    results_gb = {}
    dimensions = [2, 3]

    for dim in dimensions:
        print(f"\n--- {dim}D DISPLACEMENT TENSOR MEMORY (dim={dim}) ---")
        for dtype_name, dtype in DATA_TYPES.items():
            print(f"  Data type: {dtype_name} ({np.dtype(dtype).itemsize} bytes/elem)")
            for N in DEFAULT_N_VALUES:
                mem_bytes = calculate_memory(N, dtype, dim=dim)
                mem_mb = mem_bytes / (1024 ** 2)
                mem_gb = mem_bytes / (1024 ** 3)
                results_mb[(dim, dtype_name, N)] = mem_mb
                results_gb[(dim, dtype_name, N)] = mem_gb
                print(f"    N = {N:6d} | Memory = {mem_mb:10.2f} MB ({mem_gb:8.4f} GB)")

    # 2. NumPy nbytes Verification
    print("\n" + "=" * 78)
    print("NUMPY nbytes VERIFICATION (N = 1,000)")
    print("=" * 78)
    N_test = 1000
    verification_passed = True

    print(f"{'Dim':<5} | {'Dtype':<8} | {'Theoretical (bytes)':<20} | {'NumPy nbytes':<16} | {'Match':<6}")
    print("-" * 65)

    for dim in dimensions:
        for dtype_name, dtype in DATA_TYPES.items():
            arr = np.empty((N_test, N_test, dim), dtype=dtype)
            theoretical = calculate_memory(N_test, dtype, dim=dim)
            actual = int(arr.nbytes)
            match = (theoretical == actual)
            if not match:
                verification_passed = False
            print(f"{dim}D    | {dtype_name:<8} | {theoretical:<20,d} | {actual:<16,d} | {str(match):<6}")
            del arr

    # 3. Out-of-Memory (OOM) Limit Analysis (N ~ 50,000)
    N_oom = 50000
    print("\n" + "=" * 78)
    print(f"OUT-OF-MEMORY (OOM) LIMIT ANALYSIS (N = {N_oom:,})")
    print("=" * 78)
    print(f"At N = {N_oom:,}, dense pairwise displacement arrays reach critical scale:")

    oom_data = {}
    for dim in dimensions:
        for dtype_name, dtype in DATA_TYPES.items():
            mem_b = calculate_memory(N_oom, dtype, dim=dim)
            mem_gib = mem_b / (1024 ** 3)
            mem_gb_dec = (N_oom * N_oom * dim * np.dtype(dtype).itemsize) / 1e9
            oom_data[(dim, dtype_name)] = (mem_gib, mem_gb_dec)
            print(f"  {dim}D {dtype_name:<7} : {mem_gib:7.2f} GiB ({mem_gb_dec:6.1f} GB decimal)")

    print("\n[NOTE] A typical modern workstation has 16 GB to 32 GB of RAM.")
    print(f"  - 3D float64 at N = {N_oom:,} requires {oom_data[(3, 'float64')][0]:.2f} GiB RAM for a SINGLE array,")
    print("    guaranteeing an immediate OS Out-Of-Memory (OOM) crash or heavy swap thrashing.")
    print(f"  - Even 2D float32 requires {oom_data[(2, 'float32')][0]:.2f} GiB, exceeding consumer-grade RAM limits.")
    print("  - This fundamentally proves why O(N^2) dense broadcasting must be replaced")
    print("    by O(N) spatial domain decomposition (cell lists / neighbor lists) in production MD.")

    # 4. Generate Visual Plots for both 2D and 3D
    N_plot = np.logspace(2, 4.5, 100)

    for dim in dimensions:
        dim_suffix = f"_{dim}d" if dim == 3 else ""

        # Graph 1: Memory Scalability Log-Log
        fig, ax = plt.subplots(figsize=(8, 6))
        for dtype_name, dtype in DATA_TYPES.items():
            mem_curve_mb = (N_plot ** 2 * dim * np.dtype(dtype).itemsize) / (1024 ** 2)
            ax.loglog(N_plot, mem_curve_mb, label=f"{dtype_name} ({dim}D)", linewidth=2.2)

        # Highlight benchmarks
        for dtype_name in DATA_TYPES:
            bench_pts = [results_mb[(dim, dtype_name, N)] for N in DEFAULT_N_VALUES]
            ax.plot(DEFAULT_N_VALUES, bench_pts, "o", markersize=6)

        # Highlight OOM threshold (~16 GB)
        ax.axhline(16 * 1024, color="crimson", linestyle="--", linewidth=1.5, label="16 GB RAM Limit")
        ax.axhline(32 * 1024, color="darkred", linestyle=":", linewidth=1.5, label="32 GB RAM Limit")

        ax.set_title(f"Pairwise Displacement Tensor Memory Scaling ({dim}D)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Number of Particles (N)", fontsize=11)
        ax.set_ylabel("Memory Consumption (MB)", fontsize=11)
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend(loc="upper left", fontsize=10)
        plt.tight_layout()

        p_scale = save_dir / f"task7_memory_scalability{dim_suffix}.png"
        plt.savefig(p_scale, dpi=300)
        plt.close(fig)

        # Graph 2: Bar Comparison
        fig, ax = plt.subplots(figsize=(9, 5.5))
        bar_width = 0.35
        x_indices = np.arange(len(DEFAULT_N_VALUES))

        f32_vals = [results_mb[(dim, "float32", N)] for N in DEFAULT_N_VALUES]
        f64_vals = [results_mb[(dim, "float64", N)] for N in DEFAULT_N_VALUES]

        ax.bar(x_indices - bar_width / 2, f32_vals, bar_width, label="float32", color="#1f77b4", alpha=0.85)
        ax.bar(x_indices + bar_width / 2, f64_vals, bar_width, label="float64 (2x Memory)", color="#ff7f0e", alpha=0.85)

        ax.set_title(f"Memory Footprint by Precision: float32 vs float64 ({dim}D)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Number of Particles (N)", fontsize=11)
        ax.set_ylabel("Memory (MB)", fontsize=11)
        ax.set_xticks(x_indices)
        ax.set_xticklabels([f"N={N:,}" for N in DEFAULT_N_VALUES])
        ax.grid(True, linestyle="--", alpha=0.5, axis="y")
        ax.legend(fontsize=10)
        plt.tight_layout()

        p_prec = save_dir / f"task7_precision_comparison{dim_suffix}.png"
        plt.savefig(p_prec, dpi=300)
        plt.close(fig)

    print(f"\nGraphs saved successfully in {save_dir}:")
    print("  - task7_memory_scalability_3d.png")
    print("  - task7_precision_comparison_3d.png")
    print("  - task7_memory_scalability.png")
    print("  - task7_precision_comparison.png")
    print("Task 7 execution completed successfully.\n")

    return True


if __name__ == "__main__":
    run_task_7()
