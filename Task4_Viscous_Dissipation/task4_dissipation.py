"""
Task 4: Viscous Dissipation & Steady-State Relaxation
=====================================================
Investigates the effect of viscous friction force F_diss = -gamma * v:
  - Scans damping coefficients gamma in [0.0, 0.05, 0.1, 0.5] s^-1
  - Solves equation of motion m * a = F_spring - gamma * v
  - Tracks energy dissipation: dE/dt = -gamma * sum_i (m_i * ||v_i||^2) <= 0
  - Measures relaxation time t_relax to static mechanical equilibrium
  - Generates energy dissipation, relaxation curves, and phase-space trajectories
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
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "task4_results"


def find_relaxation_times(t, E, v_mean, dt, window=2.0, tol_E=1e-3, tol_v=1e-3):
    """
    Finds the relaxation time t_relax when the system enters steady-state equilibrium:
      1. Speed criterion: average velocity <|v|> < tol_v over a sustained time window.
      2. Energy criterion: |dE/dt| / E0 < tol_E.
    """
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


def execute_task_4(output_dir=None, show_plots=False):
    """
    Executes Task 4 dissipation experiments:
      - Simulates gamma in [0.0, 0.05, 0.1, 0.5]
      - Computes relaxation times
      - Generates summary CSV and 3 analytical plots
    """
    results_dir = Path(output_dir) if output_dir is not None else DEFAULT_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    gammas = [0.0, 0.05, 0.1, 0.5]
    dt_task4 = 0.005
    T_task4 = 30.0
    N = 12
    L = 10.0
    diss_results = {}

    print("=" * 94)
    print(f"TASK 4: DISSIPATION AND RELAXATION TO STEADY STATE (verlet, dt = {dt_task4}, seed 42)")
    print("=" * 94)
    print(f"{'gamma':<7} | {'E(0)':<8} | {'E_final':<8} | {'K_final':<10} | {'U_final':<8} | {'<v>_final':<10} | {'t_relax(<v><1e-3)':<18} | {'t_relax(dE/E0<1e-3)':<19}")
    print("-" * 94)

    csv_lines = ["gamma,E0,E_final,K_final,U_final,v_mean_final,t_relax_speed_s,t_relax_energy_s"]

    for g in gammas:
        res = run_sim(
            dt_task4, T_task4, method="verlet", seed=42, gamma=g, save_pos=True,
            k=15.0, r0=1.5, rc=1.5, L=L, N=N
        )
        diss_results[g] = res
        tv, tE = find_relaxation_times(res["t"], res["E"], res["v_mean"], dt_task4)
        tv_s = f"{tv:.2f} s" if tv is not None else "Not reached"
        tE_s = f"{tE:.2f} s" if tE is not None else "Not reached"

        print(f"{g:<7.2f} | {res['E'][0]:<8.4f} | {res['E'][-1]:<8.4f} | {res['K'][-1]:<10.2e} | "
              f"{res['U'][-1]:<8.4f} | {res['v_mean'][-1]:<10.2e} | {tv_s:<18} | {tE_s:<19}")

        csv_lines.append(f"{g},{res['E'][0]:.6f},{res['E'][-1]:.6f},{res['K'][-1]:.6e},{res['U'][-1]:.6e},{res['v_mean'][-1]:.6e},{tv_s},{tE_s}")

    print("-" * 94)
    summary_csv = results_dir / "task4_summary_table.csv"
    summary_csv.write_text("\n".join(csv_lines), encoding="utf-8")

    # -------------------------------------------------------------------------
    # PLOT 1: task4_dissipation_KUE.png
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), sharex=True)
    axes = axes.ravel()
    for idx, g in enumerate(gammas):
        r = diss_results[g]
        ax = axes[idx]
        ax.plot(r["t"], r["K"], label="K(t)", lw=1.2)
        ax.plot(r["t"], r["U"], label="U(t)", lw=1.2)
        ax.plot(r["t"], r["E"], label="E_tot(t)", lw=1.8)
        ax.set_title(f"gamma = {g} (E_final = {r['E'][-1]:.3f})")
        ax.set_xlabel("t"); ax.set_ylabel("Energy")
        ax.grid(True, alpha=0.35)
        if idx == 0:
            ax.legend(loc="upper right")

    plt.tight_layout()
    p1 = results_dir / "task4_dissipation_KUE.png"
    plt.savefig(p1, dpi=200)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # PLOT 2: task4_relaxation_curves.png
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for g in gammas:
        r = diss_results[g]
        ax1.plot(r["t"], r["E"], lw=1.6, label=f"gamma={g}")
        ax2.semilogy(r["t"], np.maximum(r["v_mean"], 1e-16), lw=1.6, label=f"gamma={g}")

    ax2.axhline(1e-3, color="red", ls="--", lw=1.1, label="threshold <v> = 1e-3")
    ax1.set_title("Total Mechanical Energy E_tot(t) vs. gamma")
    ax1.set_xlabel("t"); ax1.set_ylabel("E_tot(t)")
    ax1.grid(True, alpha=0.35); ax1.legend()

    ax2.set_title("Average Particle Speed <|v(t)|> vs. gamma")
    ax2.set_xlabel("t"); ax2.set_ylabel("<|v(t)|> (log scale)")
    ax2.grid(True, which="both", alpha=0.35); ax2.legend()

    plt.tight_layout()
    p2 = results_dir / "task4_relaxation_curves.png"
    plt.savefig(p2, dpi=200)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # PLOT 3: task4_trajectories_comparison.png
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.2))
    pos0 = diss_results[0.0]["pos_hist"]
    pos5 = diss_results[0.5]["pos_hist"]

    for i in range(N):
        ax1.plot(pos0[:, i, 0], pos0[:, i, 1], alpha=0.55, lw=1.0)
        ax2.plot(pos5[:, i, 0], pos5[:, i, 1], alpha=0.75, lw=1.3)

    ax1.scatter(pos0[-1, :, 0], pos0[-1, :, 1], c="black", s=35, zorder=5, label="Final (t=30)")
    ax2.scatter(pos5[0, :, 0], pos5[0, :, 1], c="gray", marker="x", s=40, label="Initial (t=0)")
    ax2.scatter(pos5[-1, :, 0], pos5[-1, :, 1], c="red", s=40, zorder=5, label="Arrested Final State")

    ax1.set_title("Conservative Trajectories (gamma = 0.0)")
    ax2.set_title("Dissipative Relaxation Trajectories (gamma = 0.5)")
    for ax in (ax1, ax2):
        ax.set_xlim(0, L); ax.set_ylim(0, L)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.35)
        ax.legend(loc="upper right", fontsize=8.5)

    plt.tight_layout()
    p3 = results_dir / "task4_trajectories_comparison.png"
    plt.savefig(p3, dpi=200)
    plt.close(fig)

    print(f"Saved Task 4 outputs in: {results_dir}")
    print(f"  - task4_summary_table.csv\n  - {p1.name}\n  - {p2.name}\n  - {p3.name}\n")
    return True


if __name__ == "__main__":
    execute_task_4()
