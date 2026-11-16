"""Task 5: float32 versus float64 precision study in 2D."""

from pathlib import Path
from time import perf_counter
import csv
import platform
import sys

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.physics import (
    pairwise_distances,
    pair_forces,
    walls_reflect,
    kinetic_energy,
    potential_energy,
    step_verlet,
)

# Simulation parameters
N = 12
DIM = 2
L = 10.0
K_SPRING = 15.0
R0 = 1.5
RC = 1.5
DT = 0.005
TOTAL_TIME = 10.0
SEED = 42
TIMING_REPEATS = 5

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Generate reproducible initial state
def make_initial_state():
    rng = np.random.default_rng(SEED)
    pos = rng.uniform(0.8, L - 0.8, size=(N, DIM)).astype(np.float64)
    vel = rng.uniform(-4.4, 4.4, size=(N, DIM)).astype(np.float64)
    mass = np.ones(N, dtype=np.float64)
    return pos, vel, mass


POS0_64, VEL0_64, MASS0_64 = make_initial_state()


# Cast state to target dtype
def state_for_dtype(dtype):
    return (
        POS0_64.astype(dtype),
        VEL0_64.astype(dtype),
        MASS0_64.astype(dtype),
    )


# Run simulation for specified dtype
def simulate(dtype, save_positions=False):
    pos, vel, mass = state_for_dtype(dtype)
    dt_value = dtype(DT)
    n_steps = int(round(TOTAL_TIME / DT))

    t = np.linspace(0.0, n_steps * DT, n_steps + 1, dtype=np.float64)
    K_hist = np.empty(n_steps + 1, dtype=dtype)
    U_hist = np.empty(n_steps + 1, dtype=dtype)
    E_hist = np.empty(n_steps + 1, dtype=dtype)
    pos_hist = np.empty((n_steps + 1, N, DIM), dtype=dtype) if save_positions else None

    K_hist[0] = kinetic_energy(vel, mass)
    U_hist[0] = potential_energy(pos, k=K_SPRING, r0=R0, rc=RC)
    E_hist[0] = K_hist[0] + U_hist[0]
    if save_positions:
        pos_hist[0] = pos.copy()

    F_prev = None
    for step in range(1, n_steps + 1):
        pos, vel, F_prev = step_verlet(
            pos, vel, mass, dt_value, L=L, gamma=0.0, k=K_SPRING, r0=R0, rc=RC, F_prev=F_prev
        )
        K_hist[step] = kinetic_energy(vel, mass)
        U_hist[step] = potential_energy(pos, k=K_SPRING, r0=R0, rc=RC)
        E_hist[step] = K_hist[step] + U_hist[step]
        if save_positions:
            pos_hist[step] = pos.copy()

    energy_drift = np.abs(E_hist - E_hist[0]) / E_hist[0]
    return {
        "t": t,
        "K": K_hist,
        "U": U_hist,
        "E": E_hist,
        "energy_drift": energy_drift,
        "pos_hist": pos_hist,
    }


# Track memory for state and working arrays
def compute_tracked_memory(dtype):
    itemsize = np.dtype(dtype).itemsize
    state_bytes = (N * DIM + N * DIM + N) * itemsize
    pair_bytes = (N * N * DIM * itemsize) + (N * N * itemsize) + (N * N)
    total_bytes = state_bytes + pair_bytes
    return state_bytes, pair_bytes, total_bytes


# Time repeated runs
def time_simulation(dtype, repeats=TIMING_REPEATS):
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        simulate(dtype, save_positions=False)
        samples.append(perf_counter() - start)
    return samples


# Main execution
def run_task_5():
    state32, pair32, total32 = compute_tracked_memory(np.float32)
    state64, pair64, total64 = compute_tracked_memory(np.float64)

    samples32 = time_simulation(np.float32)
    samples64 = time_simulation(np.float64)
    time32 = float(np.median(samples32))
    time64 = float(np.median(samples64))

    result32 = simulate(np.float32, save_positions=True)
    result64 = simulate(np.float64, save_positions=True)

    delta_pos = result32["pos_hist"].astype(np.float64) - result64["pos_hist"].astype(np.float64)
    rms_displacement = np.sqrt(np.mean(np.sum(delta_pos ** 2, axis=2), axis=1))
    position_error = rms_displacement / L

    max_position_error = float(np.max(position_error))
    final_position_error = float(position_error[-1])
    position_history_bytes32 = int(result32["pos_hist"].nbytes)
    position_history_bytes64 = int(result64["pos_hist"].nbytes)

    # Save metrics CSV
    metrics_path = OUT_DIR / "task5_precision_metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "float32", "float64", "ratio_or_note"])
        writer.writerow(["state_memory_bytes", state32, state64, f"{state64 / state32:.4f}"])
        writer.writerow(["pair_memory_bytes", pair32, pair64, f"{pair64 / pair32:.4f}"])
        writer.writerow(["total_tracked_memory_bytes", total32, total64, f"{total64 / total32:.4f}"])
        writer.writerow(["median_runtime_seconds", f"{time32:.6f}", f"{time64:.6f}", f"{time32 / time64:.4f}"])
        writer.writerow(["initial_total_energy", f"{float(result32['E'][0]):.8g}", f"{float(result64['E'][0]):.8g}", "exact initial state"])
        writer.writerow(["final_total_energy", f"{float(result32['E'][-1]):.8g}", f"{float(result64['E'][-1]):.8g}", "at T=10s"])
        writer.writerow(["max_relative_energy_drift", f"{float(np.max(result32['energy_drift'])):.6e}", f"{float(np.max(result64['energy_drift'])):.6e}", "bounded oscillation"])
        writer.writerow(["final_relative_position_error", f"{final_position_error:.6e}", "reference", "RMS error relative to L"])
        writer.writerow(["max_relative_position_error", f"{max_position_error:.6e}", "reference", "RMS error relative to L"])

    # Save timeseries CSV
    timeseries_path = OUT_DIR / "task5_precision_timeseries.csv"
    with open(timeseries_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "time", "K_float32", "U_float32", "E_float32", "energy_drift_float32", "K_float64", "U_float64", "E_float64", "energy_drift_float64", "relative_position_difference"])
        for i, t_val in enumerate(result64["t"]):
            writer.writerow([
                i, f"{t_val:.4f}",
                f"{float(result32['K'][i]):.8g}", f"{float(result32['U'][i]):.8g}", f"{float(result32['E'][i]):.8g}", f"{float(result32['energy_drift'][i]):.6e}",
                f"{float(result64['K'][i]):.8g}", f"{float(result64['U'][i]):.8g}", f"{float(result64['E'][i]):.8g}", f"{float(result64['energy_drift'][i]):.6e}",
                f"{float(position_error[i]):.6e}",
            ])

    # Save timing samples CSV
    timing_path = OUT_DIR / "task5_timing_samples.csv"
    with open(timing_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["repeat", "float32_wall_seconds", "float64_wall_seconds"])
        for idx, (s32, s64) in enumerate(zip(samples32, samples64), start=1):
            writer.writerow([idx, f"{s32:.6f}", f"{s64:.6f}"])

    # Plot 1: Energy comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    series = [("K", "Kinetic energy", ":"), ("U", "Potential energy", "-."), ("E", "Total energy", "-")]
    for key, label, style in series:
        ax.plot(result64["t"], result64[key].astype(np.float64), label=f"{label} - float64", lw=1.3, ls=style)
        ax.plot(result32["t"], result32[key].astype(np.float64), label=f"{label} - float32", lw=1.0, ls="--")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy (J)")
    ax.set_title("Task 5: Energy histories (float32 vs float64)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "task5_energy_comparison.png", dpi=200)
    plt.close(fig)

    # Plot 2: Energy drift and position discrepancy
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    axes[0].semilogy(result64["t"], np.maximum(result64["energy_drift"], 1e-18), label="float64")
    axes[0].semilogy(result32["t"], np.maximum(result32["energy_drift"], 1e-18), label="float32", ls="--")
    axes[0].set_ylabel("Relative energy drift")
    axes[0].set_title("Total-energy drift")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    axes[1].semilogy(result32["t"], np.maximum(position_error, 1e-18))
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Relative position error")
    axes[1].set_title("Trajectory divergence (RMS error / L)")
    axes[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "task5_position_error.png", dpi=200)
    plt.close(fig)

    # Plot 3: Runtime comparison
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(1, TIMING_REPEATS + 1)
    width = 0.35
    ax.bar(x - width / 2, samples32, width, label="float32")
    ax.bar(x + width / 2, samples64, width, label="float64")
    ax.set_xlabel("Timing repeat")
    ax.set_ylabel("Wall time (s)")
    ax.set_title("Task 5: Runtime comparison")
    ax.set_xticks(x)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "task5_runtime_comparison.png", dpi=200)
    plt.close(fig)

    # Environment specs
    env_path = OUT_DIR / "task5_environment.txt"
    env_path.write_text(
        f"Python: {sys.version.split()[0]}\nNumPy: {np.__version__}\nMatplotlib: {matplotlib.__version__}\nPlatform: {platform.platform()}\n",
        encoding="utf-8",
    )

    # Mirror to task5_2d_results for backward compatibility
    legacy_dir = SCRIPT_DIR / "task5_2d_results"
    if OUT_DIR != legacy_dir:
        legacy_dir.mkdir(parents=True, exist_ok=True)
        for f in OUT_DIR.glob("*"):
            if f.is_file():
                import shutil
                shutil.copy2(f, legacy_dir / f.name)

    print(f"Task 5 outputs saved in: {OUT_DIR}")
    return True


main = run_task_5

if __name__ == "__main__":
    run_task_5()
