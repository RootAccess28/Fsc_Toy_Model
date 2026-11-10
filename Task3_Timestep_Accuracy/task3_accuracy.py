"""
Task 3: Energy Conservation & Timestep Accuracy
===============================================
Systematic investigation of numerical accuracy and energy conservation across
timesteps dt in [0.05, 0.02, 0.01, 0.005, 0.002, 0.001] s comparing:
1. Euler (Euler-Cromer / Forward Euler) - exhibits secular energy drift O(dt).
2. Velocity Verlet - symplectic integrator bounding energy oscillations O(dt^2).
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

from core.physics import run_sim

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "task3_results"


def execute_task_3(output_dir=None, show_plots=False):
    """
    Executes Task 3 experiments:
      - Scans timesteps dt in [0.05, 0.02, 0.01, 0.005, 0.002, 0.001] s
      - 10-seed ensemble [42..51] for statistical rigor
      - Measures maximum and median relative energy errors
      - Saves convergence and energy evolution plots to output_dir
    """
    results_dir = Path(output_dir) if output_dir is not None else DEFAULT_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    dts = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
    T_task3 = 10.0

    seed42_euler = {}
    seed42_verlet = {}

    euler_med, euler_worst = [], []
    verlet_med, verlet_worst = [], []

    print("=" * 86)
    print("TASK 3: ENERGY AND TIMESTEP ACCURACY (SEED 42 & 10-SEED ENSEMBLE)")
    print("=" * 86)
    print(f"{'dt':<8} | {'Method':<8} | {'E(0) [s42]':<11} | {'eps_max (s42)':<14} | {'median (10s)':<14} | {'worst (10s)':<14}")
    print("-" * 86)

    csv_lines = ["dt,method,E0_seed42,eps_E_max_seed42,eps_E_median_10seeds,eps_E_worst_10seeds"]

    for dt in dts:
        e_max_list = []
        v_max_list = []
        for s in seeds:
            res_e = run_sim(dt, T_task3, method="euler", seed=s, k=15.0, r0=1.5, rc=1.5, L=10.0, N=12)
            res_v = run_sim(dt, T_task3, method="verlet", seed=s, k=15.0, r0=1.5, rc=1.5, L=10.0, N=12)
            e_max_list.append(res_e["eps_E_max"])
            v_max_list.append(res_v["eps_E_max"])
            if s == 42:
                seed42_euler[dt] = res_e
                seed42_verlet[dt] = res_v

        e_med, e_wst = float(np.median(e_max_list)), float(np.max(e_max_list))
        v_med, v_wst = float(np.median(v_max_list)), float(np.max(v_max_list))
        euler_med.append(e_med); euler_worst.append(e_wst)
        verlet_med.append(v_med); verlet_worst.append(v_wst)

        E0_s42 = seed42_euler[dt]["E"][0]
        print(f"{dt:<8.3f} | {'euler':<8} | {E0_s42:<11.4f} | {seed42_euler[dt]['eps_E_max']:<14.4e} | {e_med:<14.4e} | {e_wst:<14.4e}")
        print(f"{dt:<8.3f} | {'verlet':<8} | {E0_s42:<11.4f} | {seed42_verlet[dt]['eps_E_max']:<14.4e} | {v_med:<14.4e} | {v_wst:<14.4e}")
        print("-" * 86)

        csv_lines.append(f"{dt},euler,{E0_s42:.6f},{seed42_euler[dt]['eps_E_max']:.6e},{e_med:.6e},{e_wst:.6e}")
        csv_lines.append(f"{dt},verlet,{E0_s42:.6f},{seed42_verlet[dt]['eps_E_max']:.6e},{v_med:.6e},{v_wst:.6e}")

    summary_csv = results_dir / "task3_summary_table.csv"
    summary_csv.write_text("\n".join(csv_lines), encoding="utf-8")

    # -------------------------------------------------------------------------
    # PLOT 1: task3_convergence.png
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    dt_arr = np.array(dts)
    ax.loglog(dt_arr, euler_med, "o-", color="#1f77b4", lw=1.8, ms=7, label="euler, median of 10 seeds")
    ax.loglog(dt_arr, euler_worst, "o--", color="#ff7f0e", alpha=0.45, lw=1.8, ms=7, label="euler, worst of 10")
    ax.loglog(dt_arr, verlet_med, "s-", color="#2ca02c", lw=1.8, ms=7, label="verlet, median of 10 seeds")
    ax.loglog(dt_arr, verlet_worst, "s--", color="#d62728", alpha=0.45, lw=1.8, ms=7, label="verlet, worst of 10")

    # Reference slopes 1 and 2
    slope1 = 3.0 * dt_arr
    slope2 = 150.0 * (dt_arr ** 2)
    ax.loglog(dt_arr, slope1, "k--", lw=0.9, label="slope 1")
    ax.loglog(dt_arr, slope2, "k:", lw=0.9, label="slope 2")

    # 1% error line
    ax.axhline(0.01, color="red", lw=1.0)
    ax.text(0.0011, 0.0115, "1 %", color="red", fontsize=11)

    ax.set_xlabel("dt", fontsize=11)
    ax.set_ylabel("eps_E_max", fontsize=11)
    ax.set_ylim(1e-4, 1e0)
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(loc="upper left", fontsize=8.5)
    plt.tight_layout()
    p1 = results_dir / "task3_convergence.png"
    plt.savefig(p1, dpi=200)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # PLOT 2: task3_Etot_curves.png
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for dt in dts:
        step_stride = max(1, len(seed42_euler[dt]["t"]) // 250)
        ax1.plot(seed42_euler[dt]["t"][::step_stride], seed42_euler[dt]["E"][::step_stride], lw=1.1, label=f"dt={dt}")
        ax2.plot(seed42_verlet[dt]["t"][::step_stride], seed42_verlet[dt]["E"][::step_stride], lw=1.1, label=f"dt={dt}")

    ax1.set_title("euler: E_tot(t), seed 42")
    ax2.set_title("verlet: E_tot(t), seed 42")
    ax1.set_xlabel("t"); ax2.set_xlabel("t")
    ax1.grid(True, alpha=0.35); ax2.grid(True, alpha=0.35)
    ax1.legend(loc="upper left", fontsize=8); ax2.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    p2 = results_dir / "task3_Etot_curves.png"
    plt.savefig(p2, dpi=200)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # PLOT 3: task3_Etot_error_vs_dt.png
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for dt in dts:
        step_stride = max(1, len(seed42_euler[dt]["t"]) // 250)
        err_e = np.maximum(seed42_euler[dt]["rel_err"][::step_stride], 1e-16)
        err_v = np.maximum(seed42_verlet[dt]["rel_err"][::step_stride], 1e-16)
        ax1.semilogy(seed42_euler[dt]["t"][::step_stride], err_e, lw=1.0, label=f"dt={dt}")
        ax2.semilogy(seed42_verlet[dt]["t"][::step_stride], err_v, lw=1.0, label=f"dt={dt}")

    ax1.set_title("euler: |E(t)-E(0)|/E(0), seed 42")
    ax2.set_title("verlet: |E(t)-E(0)|/E(0), seed 42")
    ax1.set_xlabel("t"); ax2.set_xlabel("t")
    ax1.grid(True, alpha=0.35); ax2.grid(True, alpha=0.35)
    ax1.legend(loc="lower right", fontsize=8); ax2.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    p3 = results_dir / "task3_Etot_error_vs_dt.png"
    plt.savefig(p3, dpi=200)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # PLOT 4: task3_KUE_dt0.01.png
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 5.6), sharex=True)
    re_01 = seed42_euler[0.01]
    rv_01 = seed42_verlet[0.01]

    axes[0, 0].plot(re_01["t"], re_01["K"], lw=1.0); axes[0, 0].set_title("euler, dt=0.01: K(t)")
    axes[0, 1].plot(re_01["t"], re_01["U"], lw=1.0); axes[0, 1].set_title("euler, dt=0.01: U(t)")
    axes[0, 2].plot(re_01["t"], re_01["E"], lw=1.0); axes[0, 2].set_title("euler, dt=0.01: E_tot(t)")

    axes[1, 0].plot(rv_01["t"], rv_01["K"], lw=1.0); axes[1, 0].set_title("verlet, dt=0.01: K(t)")
    axes[1, 1].plot(rv_01["t"], rv_01["U"], lw=1.0); axes[1, 1].set_title("verlet, dt=0.01: U(t)")
    axes[1, 2].plot(rv_01["t"], rv_01["E"], lw=1.0); axes[1, 2].set_title("verlet, dt=0.01: E_tot(t)")

    for ax in axes.ravel():
        ax.grid(True, alpha=0.35)
        ax.set_xlabel("t")

    plt.tight_layout()
    p4 = results_dir / "task3_KUE_dt0.01.png"
    plt.savefig(p4, dpi=200)
    plt.close(fig)

    print(f"Saved Task 3 outputs in: {results_dir}")
    print(f"  - task3_summary_table.csv\n  - {p1.name}\n  - {p2.name}\n  - {p3.name}\n  - {p4.name}\n")
    return True


if __name__ == "__main__":
    execute_task_3()
