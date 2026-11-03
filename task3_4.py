import os
from pathlib import Path
import numpy as np
import matplotlib
if __name__ == "__main__":
    matplotlib.use("Agg")  # Guarantees figures save cleanly without GUI popups when run directly
import matplotlib.pyplot as plt

# =============================================================================
# 1. GUARANTEED DIRECTORY SETUP (NEXT TO THIS SCRIPT)
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
TASK3_DIR = SCRIPT_DIR / "task_3 results"
TASK4_DIR = SCRIPT_DIR / "task_4 results"

def setup_output_directories(verbose=True):
    TASK3_DIR.mkdir(parents=True, exist_ok=True)
    TASK4_DIR.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"Script location     : {SCRIPT_DIR}")
        print(f"Task 3 output folder: {TASK3_DIR}")
        print(f"Task 4 output folder: {TASK4_DIR}\n")

# =============================================================================
# 2. MODEL PARAMETERS (MATCHING REFERENCE PLOTS: E(0) ~ 77.9 FOR SEED 42)
# =============================================================================
N = 12              # Number of particles
L = 10.0            # Box side length
k = 15.0            # Spring / repulsion stiffness
rc = 1.5            # Interaction cutoff distance
r0 = 1.5            # Equilibrium distance = rc so U(rc) = 0 (continuous at cutoff)
T_task3 = 10.0      # Total time t = 10 (matches reference plots)
T_task4 = 30.0      # Total time for Task 4 dissipation study

def init_state(seed=42, N=N, L=L, v_max=4.4, margin=0.8):
    """Initializes positions, velocities, and masses reproducibly for a given seed."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(margin, L - margin, size=(N, 2))
    vel = rng.uniform(-v_max, v_max, size=(N, 2))
    mass = np.ones(N, dtype=np.float64)
    return pos, vel, mass

# =============================================================================
# 3. PAIRWISE DISTANCES, FORCES, ENERGIES, AND WALL REFLECTIONS
# =============================================================================
def pairwise_distances(pos, rc=rc):
    d = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt(np.sum(d**2, axis=-1))
    neighbors = (dist > 1e-12) & (dist <= rc)
    return d, dist, neighbors

def pair_forces(pos, k=k, r0=r0, rc=rc):
    d, dist, neighbors = pairwise_distances(pos, rc)
    u = d / (dist[:, :, None] + 1e-12)
    fmag = -k * (dist - r0) * neighbors
    return np.sum(fmag[:, :, None] * u, axis=1)

def total_forces(pos, vel, gamma=0.0, k=k, r0=r0, rc=rc):
    F_spring = pair_forces(pos, k, r0, rc)
    if gamma == 0.0:
        return F_spring
    return F_spring - gamma * vel

def kinetic_energy(vel, mass):
    return 0.5 * np.sum(mass[:, None] * (vel**2))

def potential_energy(pos, k=k, r0=r0, rc=rc):
    _, dist, neighbors = pairwise_distances(pos, rc)
    u_pair = 0.5 * k * ((dist - r0)**2) * neighbors
    return 0.5 * np.sum(u_pair)

def walls_reflect(pos, vel, L=L, radius=0.0):
    for dim in range(2):
        low = pos[:, dim] < radius
        vel[low, dim] = np.abs(vel[low, dim])
        pos[low, dim] = radius

        high = pos[:, dim] > (L - radius)
        vel[high, dim] = -np.abs(vel[high, dim])
        pos[high, dim] = L - radius
    return pos, vel

def particle_collisions(pos, vel, radius=0.0):
    """Elastic hard-core collision response for finite-size circular particles."""
    if radius <= 0.0:
        return pos, vel
    N_p = len(pos)
    min_dist = 2.0 * radius
    for i in range(N_p):
        for j in range(i + 1, N_p):
            delta = pos[i] - pos[j]
            dist = np.hypot(delta[0], delta[1])
            if 0.0 < dist < min_dist:
                n = delta / dist
                overlap = min_dist - dist
                pos[i] += 0.5 * overlap * n
                pos[j] -= 0.5 * overlap * n
                v_rel = vel[i] - vel[j]
                vn = np.dot(v_rel, n)
                if vn < 0.0:
                    vel[i] -= vn * n
                    vel[j] += vn * n
    return pos, vel

# =============================================================================
# 4. TIME INTEGRATORS: EULER (SIMPLE) AND VERLET (SMOOTH)
# =============================================================================
def step_euler(pos, vel, mass, dt, L=L, gamma=0.0, k=k, r0=r0, rc=rc, radius=0.0):
    """Simple update (Euler-Cromer / Forward Euler)."""
    F = total_forces(pos, vel, gamma=gamma, k=k, r0=r0, rc=rc)
    acc = F / mass[:, None]
    pos_new = pos + vel * dt
    vel_new = vel + acc * dt
    pos_new, vel_new = walls_reflect(pos_new, vel_new, L, radius=radius)
    return pos_new, vel_new

def step_verlet(pos, vel, mass, dt, L=L, gamma=0.0, k=k, r0=r0, rc=rc, radius=0.0, F_prev=None):
    """Smooth update (Velocity Verlet)."""
    if F_prev is None:
        F_prev = total_forces(pos, vel, gamma=gamma, k=k, r0=r0, rc=rc)
    a_prev = F_prev / mass[:, None]
    vel_half = vel + 0.5 * a_prev * dt
    pos_new = pos + vel_half * dt
    pos_new, vel_half = walls_reflect(pos_new, vel_half, L, radius=radius)
    F_new = total_forces(pos_new, vel_half, gamma=gamma, k=k, r0=r0, rc=rc)
    a_new = F_new / mass[:, None]
    vel_new = vel_half + 0.5 * a_new * dt
    return pos_new, vel_new, F_new

def run_sim(dt, T_max, method="verlet", seed=42, gamma=0.0, k=k, r0=r0, rc=rc, L=L, N=N, radius=0.0, save_pos=False):
    pos, vel, mass = init_state(seed=seed, N=N, L=L)
    n_steps = int(round(T_max / dt))

    t_arr = np.linspace(0.0, n_steps * dt, n_steps + 1)
    K_arr = np.zeros(n_steps + 1)
    U_arr = np.zeros(n_steps + 1)
    E_arr = np.zeros(n_steps + 1)
    v_mean = np.zeros(n_steps + 1)
    pos_hist = np.zeros((n_steps + 1, N, 2)) if save_pos else None

    K_arr[0] = kinetic_energy(vel, mass)
    U_arr[0] = potential_energy(pos, k=k, r0=r0, rc=rc)
    E_arr[0] = K_arr[0] + U_arr[0]
    v_mean[0] = np.mean(np.linalg.norm(vel, axis=1))
    if save_pos:
        pos_hist[0] = pos.copy()

    F_prev = None
    for s in range(1, n_steps + 1):
        if method == "euler":
            pos, vel = step_euler(pos, vel, mass, dt, L=L, gamma=gamma, k=k, r0=r0, rc=rc, radius=radius)
        else:
            pos, vel, F_prev = step_verlet(pos, vel, mass, dt, L=L, gamma=gamma, k=k, r0=r0, rc=rc, radius=radius, F_prev=F_prev)

        K_arr[s] = kinetic_energy(vel, mass)
        U_arr[s] = potential_energy(pos, k=k, r0=r0, rc=rc)
        E_arr[s] = K_arr[s] + U_arr[s]
        v_mean[s] = np.mean(np.linalg.norm(vel, axis=1))
        if save_pos:
            pos_hist[s] = pos.copy()

    rel_err = np.abs(E_arr - E_arr[0]) / E_arr[0]
    eps_E_max = float(np.max(rel_err))
    eps_E_rms = float(np.sqrt(np.mean(rel_err**2)))

    return {
        "t": t_arr, "K": K_arr, "U": U_arr, "E": E_arr,
        "rel_err": rel_err, "v_mean": v_mean, "pos_hist": pos_hist,
        "eps_E_max": eps_E_max, "eps_E_rms": eps_E_rms
    }

# =============================================================================
# 5. TASK 3 EXPERIMENTS & EXACT PLOTS
# =============================================================================
def execute_task_3():
    setup_output_directories(verbose=True)
    dts = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

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
            res_e = run_sim(dt, T_task3, method="euler", seed=s)
            res_v = run_sim(dt, T_task3, method="verlet", seed=s)
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

    (TASK3_DIR / "task3_summary_table.csv").write_text("\n".join(csv_lines), encoding="utf-8")

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
    slope2 = 150.0 * (dt_arr**2)
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
    p1 = TASK3_DIR / "task3_convergence.png"
    plt.savefig(p1, dpi=200)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # PLOT 2: task3_Etot_curves.png
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for dt in dts:
        # Subsample high-frequency curves slightly for crisp plotting
        step_stride = max(1, len(seed42_euler[dt]["t"]) // 250)
        ax1.plot(seed42_euler[dt]["t"][::step_stride], seed42_euler[dt]["E"][::step_stride], lw=1.1, label=f"dt={dt}")
        ax2.plot(seed42_verlet[dt]["t"][::step_stride], seed42_verlet[dt]["E"][::step_stride], lw=1.1, label=f"dt={dt}")

    ax1.set_title("euler: E_tot(t), seed 42")
    ax2.set_title("verlet: E_tot(t), seed 42")
    ax1.set_xlabel("t"); ax2.set_xlabel("t")
    ax1.grid(True, alpha=0.35); ax2.grid(True, alpha=0.35)
    ax1.legend(loc="upper left", fontsize=8); ax2.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    p2 = TASK3_DIR / "task3_Etot_curves.png"
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
    p3 = TASK3_DIR / "task3_Etot_error_vs_dt.png"
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
    p4 = TASK3_DIR / "task3_KUE_dt0.01.png"
    plt.savefig(p4, dpi=200)
    plt.close(fig)

    print(f"Saved Task 3 plots:\n  - {p1}\n  - {p2}\n  - {p3}\n  - {p4}\n")

# =============================================================================
# 6. TASK 4 EXPERIMENTS & PLOTS (DISSIPATION & RELAXATION)
# =============================================================================
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

def execute_task_4():
    setup_output_directories(verbose=True)
    gammas = [0.0, 0.05, 0.1, 0.5]
    dt_task4 = 0.005
    diss_results = {}

    print("=" * 94)
    print(f"TASK 4: DISSIPATION AND RELAXATION TO STEADY STATE (verlet, dt = {dt_task4}, seed 42)")
    print("=" * 94)
    print(f"{'gamma':<7} | {'E(0)':<8} | {'E_final':<8} | {'K_final':<10} | {'U_final':<8} | {'<v>_final':<10} | {'t_relax(<v><1e-3)':<18} | {'t_relax(dE/E0<1e-3)':<19}")
    print("-" * 94)

    csv_lines = ["gamma,E0,E_final,K_final,U_final,v_mean_final,t_relax_speed_s,t_relax_energy_s"]

    for g in gammas:
        res = run_sim(dt_task4, T_task4, method="verlet", seed=42, gamma=g, save_pos=True)
        diss_results[g] = res
        tv, tE = find_relaxation_times(res["t"], res["E"], res["v_mean"], dt_task4)
        tv_s = f"{tv:.2f} s" if tv is not None else "Not reached"
        tE_s = f"{tE:.2f} s" if tE is not None else "Not reached"

        print(f"{g:<7.2f} | {res['E'][0]:<8.4f} | {res['E'][-1]:<8.4f} | {res['K'][-1]:<10.2e} | "
              f"{res['U'][-1]:<8.4f} | {res['v_mean'][-1]:<10.2e} | {tv_s:<18} | {tE_s:<19}")

        csv_lines.append(f"{g},{res['E'][0]:.6f},{res['E'][-1]:.6f},{res['K'][-1]:.6e},{res['U'][-1]:.6e},{res['v_mean'][-1]:.6e},{tv_s},{tE_s}")

    print("-" * 94)
    (TASK4_DIR / "task4_summary_table.csv").write_text("\n".join(csv_lines), encoding="utf-8")

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
    p1 = TASK4_DIR / "task4_dissipation_KUE.png"
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
    p2 = TASK4_DIR / "task4_relaxation_curves.png"
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
    p3 = TASK4_DIR / "task4_trajectories_comparison.png"
    plt.savefig(p3, dpi=200)
    plt.close(fig)

    print(f"Saved Task 4 plots:\n  - {p1}\n  - {p2}\n  - {p3}\n")

if __name__ == "__main__":
    execute_task_3()
    execute_task_4()
    print("ALL PLOTS AND CSV TABLES SUCCESSFULLY SAVED!")