"""Task 5: float32 versus float64 precision study for the 2D atomic toy model.

This version follows the 2D quantitative model in the uploaded task3_4.py:
N=12, L=10, k=15, r0=rc=1.5, dt=0.005, T=10, seed=42.
Only the floating-point dtype changes between the two runs.

Outputs are written to task5_2d_results/:
- task5_precision_metrics.csv
- task5_timing_samples.csv
- task5_precision_timeseries.csv
- task5_energy_comparison.png
- task5_position_error.png
- task5_runtime_comparison.png
"""
from pathlib import Path
from time import perf_counter
import csv
import platform
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Parameters copied from the uploaded task3_4.py 2D Tasks 3-4 model.
# Task 5 uses gamma=0 to compare numerical precision without dissipation.
# -----------------------------------------------------------------------------
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
OUT_DIR = SCRIPT_DIR / "task5_2d_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_initial_state():
    """Generate one float64 initial state and cast it for both experiments."""
    rng = np.random.default_rng(SEED)
    pos = rng.uniform(0.8, L - 0.8, size=(N, DIM)).astype(np.float64)
    vel = rng.uniform(-4.4, 4.4, size=(N, DIM)).astype(np.float64)
    mass = np.ones(N, dtype=np.float64)
    return pos, vel, mass


POS0_64, VEL0_64, MASS0_64 = make_initial_state()


def state_for_dtype(dtype):
    return (
        POS0_64.astype(dtype),
        VEL0_64.astype(dtype),
        MASS0_64.astype(dtype),
    )


def pairwise_distances(pos, rc=RC):
    """2D pairwise displacement, distance, and neighbor matrices."""
    scalar = pos.dtype.type
    d = pos[:, None, :] - pos[None, :, :]
    dist_sq = np.sum(d * d, axis=-1, dtype=pos.dtype)
    dist = np.sqrt(dist_sq)
    neighbors = (dist > scalar(1e-12)) & (dist <= scalar(rc))
    return d, dist, neighbors


def pair_forces(pos):
    """Vectorized Hooke-style pair force used by the Task 3-4 script."""
    scalar = pos.dtype.type
    d, dist, neighbors = pairwise_distances(pos)
    denom = dist[:, :, None]
    unit = np.divide(
        d,
        denom,
        out=np.zeros_like(d),
        where=denom > scalar(0.0),
    )
    fmag = -scalar(K_SPRING) * (dist - scalar(R0)) * neighbors
    return np.sum(fmag[:, :, None] * unit, axis=1, dtype=pos.dtype)


def walls_reflect(pos, vel):
    """Reflect particles from the four faces of the 2D square [0,L]^2."""
    for axis in range(DIM):
        low = pos[:, axis] < pos.dtype.type(0.0)
        vel[low, axis] *= pos.dtype.type(-1.0)
        pos[low, axis] = pos.dtype.type(0.0)

        high = pos[:, axis] > pos.dtype.type(L)
        vel[high, axis] *= pos.dtype.type(-1.0)
        pos[high, axis] = pos.dtype.type(L)
    return pos, vel


def kinetic_energy(vel, mass):
    scalar = vel.dtype.type
    return scalar(0.5) * np.sum(
        mass[:, None] * (vel * vel), dtype=vel.dtype
    )


def potential_energy(pos):
    scalar = pos.dtype.type
    _, dist, neighbors = pairwise_distances(pos)
    pair_u = scalar(0.5) * scalar(K_SPRING) * (
        (dist - scalar(R0)) ** 2
    ) * neighbors
    # Distance/neighbor matrices contain both (i,j) and (j,i).
    return scalar(0.5) * np.sum(pair_u, dtype=pos.dtype)


def step_verlet(pos, vel, mass, dt_value, force_previous=None):
    """One conservative velocity-Verlet step, using the active dtype."""
    scalar = pos.dtype.type
    if force_previous is None:
        force_previous = pair_forces(pos)

    acceleration_previous = force_previous / mass[:, None]
    vel_half = vel + scalar(0.5) * acceleration_previous * dt_value
    pos_new = pos + vel_half * dt_value
    pos_new, vel_half = walls_reflect(pos_new, vel_half)

    force_new = pair_forces(pos_new)
    acceleration_new = force_new / mass[:, None]
    vel_new = vel_half + scalar(0.5) * acceleration_new * dt_value
    return pos_new, vel_new, force_new


def simulate(dtype, save_positions=False):
    """Run the same 2D conservative simulation using the requested dtype."""
    pos, vel, mass = state_for_dtype(dtype)
    dt_value = dtype(DT)
    n_steps = int(round(TOTAL_TIME / DT))

    t = np.linspace(0.0, n_steps * DT, n_steps + 1, dtype=np.float64)
    K_hist = np.empty(n_steps + 1, dtype=dtype)
    U_hist = np.empty(n_steps + 1, dtype=dtype)
    E_hist = np.empty(n_steps + 1, dtype=dtype)
    pos_hist = np.empty((n_steps + 1, N, DIM), dtype=dtype) if save_positions else None

    K_hist[0] = kinetic_energy(vel, mass)
    U_hist[0] = potential_energy(pos)
    E_hist[0] = K_hist[0] + U_hist[0]
    if save_positions:
        pos_hist[0] = pos

    force_previous = None
    for step_index in range(1, n_steps + 1):
        pos, vel, force_previous = step_verlet(
            pos, vel, mass, dt_value, force_previous
        )
        K_hist[step_index] = kinetic_energy(vel, mass)
        U_hist[step_index] = potential_energy(pos)
        E_hist[step_index] = K_hist[step_index] + U_hist[step_index]
        if save_positions:
            pos_hist[step_index] = pos

    E64 = E_hist.astype(np.float64)
    E0 = float(E64[0])
    energy_drift = np.abs(E64 - E0) / max(abs(E0), np.finfo(np.float64).tiny)
    return {
        "t": t,
        "K": K_hist,
        "U": U_hist,
        "E": E_hist,
        "energy_drift": energy_drift,
        "pos_hist": pos_hist,
        "final_pos": pos.copy(),
    }


def tracked_array_memory(dtype):
    """Count explicitly retained NumPy arrays (not temporary arrays/Python overhead)."""
    pos, vel, mass = state_for_dtype(dtype)
    force = pair_forces(pos)
    d, dist, neighbors = pairwise_distances(pos)
    state_bytes = pos.nbytes + vel.nbytes + mass.nbytes + force.nbytes
    pairwise_bytes = d.nbytes + dist.nbytes + neighbors.nbytes
    return state_bytes, pairwise_bytes, state_bytes + pairwise_bytes


def measure_runtimes():
    """Warm up both paths and collect five alternating-order wall-time samples."""
    simulate(np.float32, save_positions=False)
    simulate(np.float64, save_positions=False)
    samples = {np.float32: [], np.float64: []}
    for repeat in range(TIMING_REPEATS):
        order = (np.float32, np.float64) if repeat % 2 == 0 else (np.float64, np.float32)
        for dtype in order:
            start = perf_counter()
            simulate(dtype, save_positions=False)
            samples[dtype].append(perf_counter() - start)
    medians = {dtype: float(np.median(values)) for dtype, values in samples.items()}
    return medians, samples


def main():
    result32 = simulate(np.float32, save_positions=True)
    result64 = simulate(np.float64, save_positions=True)

    # Compare both stored trajectories in float64 so the metric arithmetic is consistent.
    p32 = result32["pos_hist"].astype(np.float64)
    p64 = result64["pos_hist"].astype(np.float64)
    denominator = np.linalg.norm(p64, axis=(1, 2))
    position_error = np.linalg.norm(p32 - p64, axis=(1, 2)) / np.maximum(
        denominator, np.finfo(np.float64).tiny
    )
    final_position_error = float(position_error[-1])
    max_position_error = float(np.max(position_error))

    runtime_medians, runtime_samples = measure_runtimes()
    time32 = runtime_medians[np.float32]
    time64 = runtime_medians[np.float64]
    samples32 = runtime_samples[np.float32]
    samples64 = runtime_samples[np.float64]

    state32, pair32, total32 = tracked_array_memory(np.float32)
    state64, pair64, total64 = tracked_array_memory(np.float64)

    # Optional stored trajectory history used for the position-error calculation.
    # This is reported separately from working arrays to keep the memory accounting clear.
    n_steps = int(round(TOTAL_TIME / DT))
    position_history_bytes32 = (n_steps + 1) * N * DIM * np.dtype(np.float32).itemsize
    position_history_bytes64 = (n_steps + 1) * N * DIM * np.dtype(np.float64).itemsize

    metrics_rows = [
        ("float32", 4, state32, pair32, total32, position_history_bytes32,
         total32 + position_history_bytes32, time32,
         float(result32["E"][0]), float(result32["E"][-1]),
         float(np.max(result32["energy_drift"])), final_position_error, max_position_error),
        ("float64", 8, state64, pair64, total64, position_history_bytes64,
         total64 + position_history_bytes64, time64,
         float(result64["E"][0]), float(result64["E"][-1]),
         float(np.max(result64["energy_drift"])), 0.0, 0.0),
    ]

    metrics_path = OUT_DIR / "task5_precision_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "precision", "bytes_per_float", "state_bytes", "pairwise_bytes",
            "tracked_total_bytes", "saved_position_history_bytes",
            "tracked_plus_position_history_bytes", "median_runtime_seconds", "E_initial",
            "E_final", "max_relative_energy_drift", "epsilon_pos_final_vs_float64",
            "epsilon_pos_max_vs_float64"
        ])
        writer.writerows(metrics_rows)

    timing_path = OUT_DIR / "task5_timing_samples.csv"
    with timing_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["repeat", "float32_seconds", "float64_seconds"])
        for i, (t32, t64) in enumerate(zip(samples32, samples64), start=1):
            writer.writerow([i, t32, t64])

    timeseries_path = OUT_DIR / "task5_precision_timeseries.csv"
    with timeseries_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time", "K_float32", "U_float32", "E_float32",
            "K_float64", "U_float64", "E_float64",
            "relative_energy_drift_float32", "relative_energy_drift_float64",
            "epsilon_pos_float32_vs_float64"
        ])
        for i, t in enumerate(result32["t"]):
            writer.writerow([
                t,
                float(result32["K"][i]), float(result32["U"][i]), float(result32["E"][i]),
                float(result64["K"][i]), float(result64["U"][i]), float(result64["E"][i]),
                float(result32["energy_drift"][i]), float(result64["energy_drift"][i]),
                float(position_error[i]),
            ])

    # Plot 1: overlay K, U, and E so the energy comparison fits a compact report figure.
    fig, ax = plt.subplots(figsize=(10, 4.3))
    series = [
        ("K", "Kinetic energy K(t)", "-"),
        ("U", "Potential energy U(t)", "-"),
        ("E", "Total energy E(t)", "-"),
    ]
    for key, label, style in series:
        ax.plot(result64["t"], result64[key].astype(np.float64),
                label=f"{label} - float64", linewidth=1.35, linestyle=style)
        ax.plot(result32["t"], result32[key].astype(np.float64),
                label=f"{label} - float32", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Time")
    ax.set_ylabel("Energy")
    ax.set_title("Task 5 - Energy histories in a 2D square box")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    energy_plot = OUT_DIR / "task5_energy_comparison.png"
    fig.savefig(energy_plot, dpi=220)
    plt.close(fig)

    # Plot 2: energy drift and relative trajectory difference.
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].semilogy(result64["t"], np.maximum(result64["energy_drift"], 1e-18), label="float64")
    axes[0].semilogy(result32["t"], np.maximum(result32["energy_drift"], 1e-18), label="float32", linestyle="--")
    axes[0].set_ylabel("Relative energy drift")
    axes[0].set_title("Total-energy drift")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()
    axes[1].semilogy(result32["t"], np.maximum(position_error, 1e-18))
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Relative position difference")
    axes[1].set_title("Trajectory difference: float32 compared with float64")
    axes[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    error_plot = OUT_DIR / "task5_position_error.png"
    fig.savefig(error_plot, dpi=220)
    plt.close(fig)

    # Plot 3: timing samples and median.
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(1, TIMING_REPEATS + 1)
    width = 0.36
    ax.bar(x - width / 2, samples32, width, label="float32")
    ax.bar(x + width / 2, samples64, width, label="float64")
    ax.set_xlabel("Timing repeat")
    ax.set_ylabel("Wall time (seconds)")
    ax.set_title("Task 5 - Runtime comparison (five runs)")
    ax.set_xticks(x)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    runtime_plot = OUT_DIR / "task5_runtime_comparison.png"
    fig.savefig(runtime_plot, dpi=220)
    plt.close(fig)

    # Record execution environment for reproducibility.
    env_path = OUT_DIR / "task5_environment.txt"
    env_path.write_text(
        f"Python: {sys.version.split()[0]}\n"
        f"NumPy: {np.__version__}\n"
        f"Matplotlib: {matplotlib.__version__}\n"
        f"Platform: {platform.platform()}\n"
        f"Processor: {platform.processor() or 'not reported by platform module'}\n",
        encoding="utf-8",
    )

    print("TASK 5 - FLOAT32 VS FLOAT64 (2D SQUARE BOX)")
    print(f"N={N}, DIM={DIM}, L={L}, k={K_SPRING}, r0={R0}, rc={RC}, dt={DT}, T={TOTAL_TIME}, seed={SEED}")
    print(f"Python={sys.version.split()[0]}, NumPy={np.__version__}, platform={platform.platform()}")
    print("Same float64-generated initial state cast to the tested dtype; gamma=0; Velocity-Verlet.")
    print()
    print(f"{'Metric':<43} {'float32':>16} {'float64':>16}")
    print("-" * 78)
    print(f"{'Tracked state-array memory (bytes)':<43} {state32:>16} {state64:>16}")
    print(f"{'Tracked pairwise-array memory (bytes)':<43} {pair32:>16} {pair64:>16}")
    print(f"{'Tracked total NumPy-array memory (bytes)':<43} {total32:>16} {total64:>16}")
    print(f"{'Median runtime (seconds)':<43} {time32:>16.6f} {time64:>16.6f}")
    print(f"{'Initial total energy':<43} {float(result32['E'][0]):>16.8g} {float(result64['E'][0]):>16.8g}")
    print(f"{'Final total energy':<43} {float(result32['E'][-1]):>16.8g} {float(result64['E'][-1]):>16.8g}")
    print(f"{'Maximum relative energy drift':<43} {float(np.max(result32['energy_drift'])):>16.6e} {float(np.max(result64['energy_drift'])):>16.6e}")
    print(f"{'Final relative position difference vs float64':<43} {final_position_error:>16.6e} {'reference':>16}")
    print(f"{'Maximum relative position difference vs float64':<43} {max_position_error:>16.6e} {'reference':>16}")
    print()
    print(f"Saved position-history memory (bytes): float32={position_history_bytes32}, float64={position_history_bytes64}")
    print(f"Working arrays + position history (bytes): float32={total32 + position_history_bytes32}, float64={total64 + position_history_bytes64}")
    print(f"float64/float32 tracked-memory ratio: {total64 / total32:.4f}")
    print(f"float32 timing samples: {[round(x, 6) for x in samples32]}")
    print(f"float64 timing samples: {[round(x, 6) for x in samples64]}")
    print(f"float32/float64 median time ratio: {time32 / time64:.4f} (less than 1 means float32 was faster)")
    print(f"Outputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
