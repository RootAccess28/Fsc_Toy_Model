"""Task 3: Energy conservation and timestep accuracy."""

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


# Execute timestep accuracy experiment across dt values
def execute_task_3(output_dir=None, show_plots=False):
    results_dir = Path(output_dir) if output_dir is not None else DEFAULT_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    dts = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
    T_task3 = 10.0

    seed42_euler = {}
    seed42_verlet = {}

    euler_med, euler_worst = [], []
    verlet_med, verlet_worst = [], []

    print("Task 3: Timestep Accuracy and Energy Conservation")
    print("-" * 72)
    print(f"{'dt (s)':<8} | {'Method':<8} | {'E(0) (J)':<10} | {'Max Rel Drift':<16} | {'Median (10 seeds)':<18}")
    print("-" * 72)

    csv_lines = ["dt,method,E0_seed42,eps_E_max_seed42,eps_E_median_10seeds,eps_E_worst_10seeds"]

    for dt in dts:
        e_max_list = []
        v_max_list = []

        for s in seeds:
            re = run_sim(dt, T_task3, method="euler", seed=s)
            rv = run_sim(dt, T_task3, method="verlet", seed=s)

            e_max_list.append(re["eps_E_max"])
            v_max_list.append(rv["eps_E_max"])

            if s == 42:
                seed42_euler[dt] = re
                seed42_verlet[dt] = rv

        med_e, worst_e = float(np.median(e_max_list)), float(np.max(e_max_list))
        med_v, worst_v = float(np.median(v_max_list)), float(np.max(v_max_list))

        euler_med.append(med_e)
        euler_worst.append(worst_e)
        verlet_med.append(med_v)
        verlet_worst.append(worst_v)

        e0_42 = seed42_euler[dt]["E"][0]
        eps_e_42 = seed42_euler[dt]["eps_E_max"]
        eps_v_42 = seed42_verlet[dt]["eps_E_max"]

        print(f"{dt:<8.3f} | {'Euler':<8} | {e0_42:<10.4f} | {eps_e_42:<16.4e} | {med_e:<18.4e}")
        print(f"{dt:<8.3f} | {'Verlet':<8} | {e0_42:<10.4f} | {eps_v_42:<16.4e} | {med_v:<18.4e}")
        print("-" * 72)

        csv_lines.append(f"{dt},euler,{e0_42:.6f},{eps_e_42:.6e},{med_e:.6e},{worst_e:.6e}")
        csv_lines.append(f"{dt},verlet,{e0_42:.6f},{eps_v_42:.6e},{med_v:.6e},{worst_v:.6e}")

    # Save CSV metrics
    csv_path = results_dir / "task3_summary_table.csv"
    with open(csv_path, "w") as f:
        f.write("\n".join(csv_lines) + "\n")

    # Plot 1: Convergence
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.loglog(dts, euler_med, "o--", color="#d62728", label="euler (median)", lw=1.5)
    ax.loglog(dts, euler_worst, "s:", color="#d62728", alpha=0.5, label="euler (worst)")
    ax.loglog(dts, verlet_med, "o-", color="#1f77b4", label="verlet (median)", lw=1.8)
    ax.loglog(dts, verlet_worst, "s:", color="#1f77b4", alpha=0.5, label="verlet (worst)")

    dt_ref = np.array([0.001, 0.05])
    ax.loglog(dt_ref, dt_ref * 40.0, "k--", alpha=0.45, label="slope 1 (O(dt))")
    ax.loglog(dt_ref, (dt_ref ** 2) * 80.0, "k-.", alpha=0.45, label="slope 2 (O(dt^2))")

    ax.set_xlabel("timestep dt (s)")
    ax.set_ylabel("relative energy drift")
    ax.set_title("Task 3: Convergence Rate")
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    p1 = results_dir / "task3_convergence.png"
    plt.savefig(p1, dpi=200)
    plt.close(fig)

    # Plot 2: Total energy evolution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2), sharey=False)
    for dt in dts:
        step_stride = max(1, len(seed42_euler[dt]["t"]) // 250)
        ax1.plot(seed42_euler[dt]["t"][::step_stride], seed42_euler[dt]["E"][::step_stride], lw=1.1, label=f"dt={dt}")
        ax2.plot(seed42_verlet[dt]["t"][::step_stride], seed42_verlet[dt]["E"][::step_stride], lw=1.1, label=f"dt={dt}")

    ax1.set_title("Euler: E_tot(t)")
    ax2.set_title("Verlet: E_tot(t)")
    ax1.set_xlabel("t (s)"); ax2.set_xlabel("t (s)")
    ax1.grid(True, alpha=0.35); ax2.grid(True, alpha=0.35)
    ax1.legend(loc="upper left", fontsize=8); ax2.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    p2 = results_dir / "task3_Etot_curves.png"
    plt.savefig(p2, dpi=200)
    plt.close(fig)

    # Plot 3: Relative error vs time
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for dt in dts:
        step_stride = max(1, len(seed42_euler[dt]["t"]) // 250)
        err_e = np.maximum(seed42_euler[dt]["rel_err"][::step_stride], 1e-16)
        err_v = np.maximum(seed42_verlet[dt]["rel_err"][::step_stride], 1e-16)
        ax1.semilogy(seed42_euler[dt]["t"][::step_stride], err_e, lw=1.0, label=f"dt={dt}")
        ax2.semilogy(seed42_verlet[dt]["t"][::step_stride], err_v, lw=1.0, label=f"dt={dt}")

    ax1.set_title("Euler: Error vs time")
    ax2.set_title("Verlet: Error vs time")
    ax1.set_xlabel("t (s)"); ax2.set_xlabel("t (s)")
    ax1.grid(True, alpha=0.35); ax2.grid(True, alpha=0.35)
    ax1.legend(loc="lower right", fontsize=8); ax2.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    p3 = results_dir / "task3_Etot_error_vs_dt.png"
    plt.savefig(p3, dpi=200)
    plt.close(fig)

    # Plot 4: Energy partition for dt=0.01
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 5.2), sharex=True)
    re_01 = seed42_euler[0.01]
    rv_01 = seed42_verlet[0.01]

    axes[0, 0].plot(re_01["t"], re_01["K"], lw=1.0); axes[0, 0].set_title("Euler dt=0.01: K(t)")
    axes[0, 1].plot(re_01["t"], re_01["U"], lw=1.0); axes[0, 1].set_title("Euler dt=0.01: U(t)")
    axes[0, 2].plot(re_01["t"], re_01["E"], lw=1.0); axes[0, 2].set_title("Euler dt=0.01: E(t)")

    axes[1, 0].plot(rv_01["t"], rv_01["K"], lw=1.0); axes[1, 0].set_title("Verlet dt=0.01: K(t)")
    axes[1, 1].plot(rv_01["t"], rv_01["U"], lw=1.0); axes[1, 1].set_title("Verlet dt=0.01: U(t)")
    axes[1, 2].plot(rv_01["t"], rv_01["E"], lw=1.0); axes[1, 2].set_title("Verlet dt=0.01: E(t)")

    for ax in axes.ravel():
        ax.grid(True, alpha=0.35)
        ax.set_xlabel("t (s)")

    plt.tight_layout()
    p4 = results_dir / "task3_KUE_dt0.01.png"
    plt.savefig(p4, dpi=200)
    plt.close(fig)

    # Mirror files to task3_results for backward compatibility
    legacy_dir = SCRIPT_DIR / "task3_results"
    if results_dir != legacy_dir:
        legacy_dir.mkdir(parents=True, exist_ok=True)
        for f in results_dir.glob("*"):
            if f.is_file():
                import shutil
                shutil.copy2(f, legacy_dir / f.name)

    print(f"Task 3 outputs saved in: {results_dir}")
    return True


if __name__ == "__main__":
    execute_task_3()
