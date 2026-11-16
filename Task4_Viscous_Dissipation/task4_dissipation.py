"""Task 4: Viscous dissipation and steady-state relaxation."""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.physics import run_sim

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"


# Determine steady-state relaxation times based on speed and energy slope
def find_relaxation_times(t, E, v_mean, dt, window=2.0, tol_E=1e-3, tol_v=1e-3):
    w_steps = int(round(window / dt))
    E0 = E[0]

    t_v = None
    for i in range(len(t) - w_steps):
        if np.all(v_mean[i:i + w_steps] < tol_v):
            t_v = float(t[i])
            break

    t_E = None
    for i in range(w_steps, len(t)):
        if (abs(E[i] - E[i - w_steps]) / E0) < tol_E and v_mean[i] < 0.05:
            t_E = float(t[i])
            break

    return t_v, t_E


# Execute viscous dissipation experiment across damping coefficients
def execute_task_4(output_dir=None, show_plots=False):
    results_dir = Path(output_dir) if output_dir is not None else DEFAULT_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    gammas = [0.0, 0.05, 0.1, 0.5]
    dt_task4 = 0.005
    T_task4 = 20.0
    seed_task4 = 42

    results = {}
    print("Task 4: Viscous Dissipation and Steady-State Relaxation")
    print("-" * 75)
    print(f"{'gamma (s^-1)':<14} | {'E(0) (J)':<10} | {'E_final (J)':<12} | {'K_final (J)':<12} | {'t_relax':<12}")
    print("-" * 75)

    csv_lines = ["gamma,E0,E_final,K_final,U_final,v_mean_final,t_relax_v,t_relax_E"]

    for g in gammas:
        res = run_sim(dt_task4, T_task4, method="verlet", seed=seed_task4, gamma=g, save_pos=True)
        results[g] = res

        t_v, t_E = find_relaxation_times(res["t"], res["E"], res["v_mean"], dt_task4)

        e0 = res["E"][0]
        ef = res["E"][-1]
        kf = res["K"][-1]
        uf = res["U"][-1]
        vf = res["v_mean"][-1]

        tv_str = f"{t_v:.2f} s" if t_v is not None else "N/A"
        tE_str = f"{t_E:.2f} s" if t_E is not None else "N/A"

        print(f"{g:<14.2f} | {e0:<10.4f} | {ef:<12.4f} | {kf:<12.4e} | {tv_str:<12}")
        csv_lines.append(f"{g},{e0:.6f},{ef:.6f},{kf:.6e},{uf:.6f},{vf:.6e},{tv_str},{tE_str}")

    print("-" * 75)

    # Save CSV metrics
    csv_path = results_dir / "task4_summary_table.csv"
    with open(csv_path, "w") as f:
        f.write("\n".join(csv_lines) + "\n")

    # Plot 1: Energy partition across gammas
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), sharex=True)
    coords = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for (r, c), g in zip(coords, gammas):
        ax = axes[r, c]
        res = results[g]
        ax.plot(res["t"], res["K"], label="K(t)", lw=1.2)
        ax.plot(res["t"], res["U"], label="U(t)", lw=1.2)
        ax.plot(res["t"], res["E"], "k--", label="E_tot(t)", lw=1.2)
        ax.set_title(f"gamma = {g}")
        ax.set_xlabel("t (s)")
        ax.set_ylabel("energy (J)")
        ax.grid(True, alpha=0.35)
        ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    p1 = results_dir / "task4_dissipation_KUE.png"
    plt.savefig(p1, dpi=200)
    plt.close(fig)

    # Plot 2: Relaxation decay curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    for g in gammas:
        ax1.semilogy(results[g]["t"], np.maximum(results[g]["E"], 1e-12), label=f"gamma={g}")
        ax2.semilogy(results[g]["t"], np.maximum(results[g]["v_mean"], 1e-12), label=f"gamma={g}")

    ax1.set_title("E_tot(t) vs gamma")
    ax2.set_title("<|v|>(t) vs gamma")
    ax1.set_xlabel("t (s)"); ax2.set_xlabel("t (s)")
    ax1.grid(True, which="both", alpha=0.35)
    ax2.grid(True, which="both", alpha=0.35)
    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    p2 = results_dir / "task4_relaxation_curves.png"
    plt.savefig(p2, dpi=200)
    plt.close(fig)

    # Plot 3: Trajectory spatial comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.8))
    pos_undamped = results[0.0]["pos_hist"]
    pos_damped = results[0.5]["pos_hist"]

    for i in range(pos_undamped.shape[1]):
        ax1.plot(pos_undamped[:, i, 0], pos_undamped[:, i, 1], lw=0.6, alpha=0.7)
        ax2.plot(pos_damped[:, i, 0], pos_damped[:, i, 1], lw=0.6, alpha=0.7)

    ax1.set_title("Trajectories (gamma = 0.0)")
    ax2.set_title("Trajectories (gamma = 0.5)")
    ax1.set_xlim(0, 10); ax1.set_ylim(0, 10)
    ax2.set_xlim(0, 10); ax2.set_ylim(0, 10)
    ax1.set_aspect("equal"); ax2.set_aspect("equal")
    plt.tight_layout()
    p3 = results_dir / "task4_trajectories_comparison.png"
    plt.savefig(p3, dpi=200)
    plt.close(fig)

    # Mirror files to task4_results for backward compatibility
    legacy_dir = SCRIPT_DIR / "task4_results"
    if results_dir != legacy_dir:
        legacy_dir.mkdir(parents=True, exist_ok=True)
        for f in results_dir.glob("*"):
            if f.is_file():
                import shutil
                shutil.copy2(f, legacy_dir / f.name)

    print(f"Task 4 outputs saved in: {results_dir}")
    return True


if __name__ == "__main__":
    execute_task_4()
