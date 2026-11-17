"""Task 8: Spatial Partitioning via Cell-Linked Lists (O(N) Scaling)."""

import os
import sys
import time
import tracemalloc
import csv
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure workspace root is in python path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.physics import (
    pairwise_distances,
    pair_forces as pair_forces_vec,
    potential_energy as potential_vec,
    walls_reflect,
    build_cells,
    neighbor_pairs_cell,
    pair_forces_cell,
    potential_energy_cell as potential_cell,
)

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

# Physical parameters matching standard benchmark configuration
K_SPRING = 15.0
R_REST = 0.8
R_CUT = 2.0
DENSITY = 12.0 / (10.0 ** 2)


# Constant number density box scaling
def box_for(n):
    return 10.0 * np.sqrt(n / 12.0)


# Random state initialization
def make_particles(n, seed=0):
    rng = np.random.default_rng(seed)
    L = box_for(n)
    pos = rng.random((n, 2)) * L
    vel = rng.standard_normal((n, 2)) * 0.5
    mass = np.ones(n)
    return pos, vel, mass, L


# Scalar Python double-loop pair forces
def pair_forces_loop(pos, k=K_SPRING, r0=R_REST, rc=R_CUT):
    n = len(pos)
    F = np.zeros_like(pos)
    d, dist, neighbors = pairwise_distances(pos, rc=rc)
    for i in range(n):
        for j in range(i + 1, n):
            if neighbors[i, j]:
                fmag = -k * (dist[i, j] - r0)
                f = fmag * d[i, j] / (dist[i, j] + 1e-14)
                F[i] += f
                F[j] -= f
    return F


# Wrapper for pair_forces_vec using local benchmark parameters
def pair_forces_vectorized(pos):
    return pair_forces_vec(pos, k=K_SPRING, r0=R_REST, rc=R_CUT)


# Wrapper for potential_vec using local benchmark parameters
def potential_vectorized(pos):
    return potential_vec(pos, k=K_SPRING, r0=R_REST, rc=R_CUT)


# Wrapper for pair_forces_cell using local benchmark parameters
def pair_forces_cell_local(pos, L):
    return pair_forces_cell(pos, L=L, k=K_SPRING, r0=R_REST, rc=R_CUT)


# Wrapper for potential_cell using local benchmark parameters
def potential_cell_local(pos, L):
    return potential_cell(pos, L=L, k=K_SPRING, r0=R_REST, rc=R_CUT)


# Velocity Verlet integrator with arbitrary force evaluation function
def run_sim(force_fn, pot_fn, pos, vel, mass, L, dt, steps):
    pos = pos.copy()
    vel = vel.copy()
    F = force_fn(pos)
    E = np.empty(steps + 1)
    E[0] = 0.5 * np.sum(mass[:, None] * vel ** 2) + pot_fn(pos)
    for s in range(steps):
        vh = vel + 0.5 * dt * F / mass[:, None]
        pos = pos + vh * dt
        walls_reflect(pos, vh, L=L)
        F = force_fn(pos)
        vel = vh + 0.5 * dt * F / mass[:, None]
        E[s + 1] = 0.5 * np.sum(mass[:, None] * vel ** 2) + pot_fn(pos)
    return pos, vel, E


# Median runtime measurement over repeated trials
def median_time(fn, repeats=5):
    fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


# Peak heap memory allocation measurement
def peak_mem(fn):
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


# Export rows to CSV
def save_csv(filename, header, rows, out_dir=OUT):
    with open(Path(out_dir) / filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


# Experiment 1: Mathematical correctness and bitwise force agreement
def exp_correctness(out_dir=OUT):
    print("\n--- EXPERIMENT 1: Force Agreement (Cell List vs All-Pairs) ---")
    rows = []
    for n in (12, 200, 1000, 3000):
        pos, _, _, L = make_particles(n, seed=1)
        Fb = pair_forces_vectorized(pos)
        Fc = pair_forces_cell_local(pos, L)
        err_abs = float(np.max(np.abs(Fb - Fc)))
        err_rel = float(np.linalg.norm(Fb - Fc) / np.linalg.norm(Fb))
        npairs_base = int(np.triu(pairwise_distances(pos, rc=R_CUT)[2], 1).sum())
        npairs_cell = len(neighbor_pairs_cell(pos, L, rc=R_CUT)[0])
        dU = float(abs(potential_vectorized(pos) - potential_cell_local(pos, L)))
        net_force = float(np.linalg.norm(Fc.sum(axis=0)))
        rows.append([n, npairs_base, npairs_cell, err_abs, err_rel, dU, net_force])
        print(f"N={n:5d} | pairs: base={npairs_base:5d}, cell={npairs_cell:5d} | "
              f"max|dF|={err_abs:.2e} N | rel={err_rel:.2e} | |sum F|={net_force:.2e} N")

    # Boundary boundary condition check
    pos_boundary = np.array([[0.0, 0.0], [0.5, 0.0], [10.0, 10.0], [9.6, 10.0], [10.0, 0.0], [5.0, 5.0]])
    e_bound = float(np.max(np.abs(pair_forces_vectorized(pos_boundary) - pair_forces_cell_local(pos_boundary, 10.0))))
    print(f"Boundary check (particles at x, y in {{0, L}}): max|dF| = {e_bound:.2e} N")

    save_csv("task8_correctness.csv",
             ["N", "pairs_baseline", "pairs_cell", "max_abs_force_diff", "rel_force_diff", "abs_U_diff", "norm_sum_F_cell"],
             rows, out_dir=out_dir)
    return rows


# Experiment 2: Execution runtime and heap memory scaling vs N
def exp_timing(fast=False, out_dir=OUT):
    print("\n--- EXPERIMENT 2: Runtime and Memory Scaling vs N ---")
    if fast:
        ns = [50, 200, 500, 1000, 2000]
        n_loop_max = 500
        n_vec_max = 2000
    else:
        ns = [50, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
        n_loop_max = 1000
        n_vec_max = 5000

    rows = []
    for n in ns:
        pos, _, _, L = make_particles(n, seed=2)
        t_cell = median_time(lambda: pair_forces_cell_local(pos, L), repeats=5)
        m_cell = peak_mem(lambda: pair_forces_cell_local(pos, L))
        t_vec = m_vec = t_loop = np.nan

        if n <= n_vec_max:
            t_vec = median_time(lambda: pair_forces_vectorized(pos), repeats=5 if n <= 2000 else 3)
            m_vec = peak_mem(lambda: pair_forces_vectorized(pos))
        if n <= n_loop_max:
            t_loop = median_time(lambda: pair_forces_loop(pos), repeats=3)

        rows.append([n, t_loop, t_vec, t_cell, m_vec, m_cell])
        sp = t_vec / t_cell if n <= n_vec_max else np.nan
        m_vec_mb = m_vec / 1e6 if np.isfinite(m_vec) else float("nan")
        m_cell_mb = m_cell / 1e6
        print(f"N={n:6d} | loop={t_loop:8.4f}s | vec={t_vec:8.4f}s | cell={t_cell:8.5f}s | "
              f"speedup={sp:6.1f}x | mem vec={m_vec_mb:7.1f} MB | cell={m_cell_mb:6.2f} MB")

    save_csv("task8_timing.csv",
             ["N", "t_python_loop_s", "t_numpy_vec_s", "t_cell_list_s", "peak_mem_vec_bytes", "peak_mem_cell_bytes"],
             rows, out_dir=out_dir)
    return np.array(rows, dtype=float)


# Experiment 3: Cell list computational bottleneck breakdown
def exp_breakdown():
    print("\n--- EXPERIMENT 3: Algorithmic Timing Breakdown (N = 20,000) ---")
    n = 20000
    pos, _, _, L = make_particles(n, seed=3)
    t_build = median_time(lambda: build_cells(pos, L, rc=R_CUT), repeats=5)
    t_pairs = median_time(lambda: neighbor_pairs_cell(pos, L, rc=R_CUT), repeats=5)
    t_total = median_time(lambda: pair_forces_cell_local(pos, L), repeats=5)
    print(f"Grid construction : {t_build * 1e3:.2f} ms")
    print(f"Stencil filtering : {t_pairs * 1e3:.2f} ms")
    print(f"Total force eval  : {t_total * 1e3:.2f} ms")
    return t_build, t_pairs, t_total


# Experiment 4: Full dynamic trajectory and energy conservation verification
def exp_simulation(fast=False):
    print("\n--- EXPERIMENT 4: Symplectic Trajectory & Energy Invariance ---")
    n = 1000 if fast else 2000
    steps = 50 if fast else 200
    dt = 0.005

    pos, vel, mass, L = make_particles(n, seed=4)
    t0 = time.perf_counter()
    pb, vb, Eb = run_sim(pair_forces_vectorized, potential_vectorized, pos, vel, mass, L, dt, steps)
    t_old = time.perf_counter() - t0

    t0 = time.perf_counter()
    pc, vc, Ec = run_sim(lambda p: pair_forces_cell_local(p, L),
                         lambda p: potential_cell_local(p, L),
                         pos, vel, mass, L, dt, steps)
    t_new = time.perf_counter() - t0

    eps_pos = float(np.linalg.norm(pb - pc) / np.linalg.norm(pb))
    drift_b = float(np.max(np.abs((Eb - Eb[0]) / Eb[0])))
    drift_c = float(np.max(np.abs((Ec - Ec[0]) / Ec[0])))
    dE_rel = float(np.max(np.abs(Eb - Ec) / abs(Eb[0])))
    inside = bool(np.all((pc >= 0) & (pc <= L)))

    print(f"Simulation steps  : {steps} steps (dt={dt} s)")
    print(f"Wall time (all-pairs) : {t_old:.2f} s")
    print(f"Wall time (cell list) : {t_new:.2f} s (speedup = {t_old / t_new:.1f}x)")
    print(f"Trajectory difference : ||r_old - r_cell|| / ||r_old|| = {eps_pos:.2e}")
    print(f"Energy drift (all-pairs): {drift_b:.3e}")
    print(f"Energy drift (cell list): {drift_c:.3e} (drift difference = {dE_rel:.2e})")
    print(f"Boundary containment  : {inside}")

    return {
        "t_old": t_old, "t_new": t_new, "eps_pos": eps_pos,
        "drift_b": drift_b, "drift_c": drift_c,
        "Eb": Eb, "Ec": Ec, "dt": dt, "N": n
    }


# Experiment 5: Pair candidate work reduction metrics
def exp_candidates(out_dir=OUT):
    print("\n--- EXPERIMENT 5: Candidate Pair Filtering Efficiency ---")
    rows = []
    for n in (200, 1000, 5000, 20000):
        pos, _, _, L = make_particles(n, seed=5)
        cx, cy, order, start, count, ncell = build_cells(pos, L, rc=R_CUT)
        cand = 0
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                nx, ny = cx + ox, cy + oy
                ok = (nx >= 0) & (nx < ncell) & (ny >= 0) & (ny < ncell)
                cand += int(np.where(ok, count[np.where(ok, nx * ncell + ny, 0)], 0).sum())
        full = n * n
        nb = len(neighbor_pairs_cell(pos, L, rc=R_CUT)[0])
        reduction = full / cand
        rows.append([n, full, cand, nb, reduction])
        print(f"N={n:6d} | All-pairs: {full:10d} | Cell candidates: {cand:7d} | "
              f"Interacting pairs: {nb:6d} | Work reduction: {reduction:6.1f}x")

    save_csv("task8_candidates.csv",
             ["N", "all_pairs_N2", "cell_candidates", "true_pairs", "work_reduction"],
             rows, out_dir=out_dir)


# Experiment 6: Verification under simulator physical configurations (Fixed Box L = 10.0)
def exp_simulator_verification(out_dir=OUT):
    print("\n--- EXPERIMENT 6: Verification under Simulator Physical Configurations ---")
    from core.physics import init_state, step_verlet
    rows = []

    # 1. Canonical Configuration (k=15.0, r0=1.5, rc=1.5, N=12, L=10.0)
    pos, vel, mass = init_state(seed=42, N=12, L=10.0, dtype=np.float64)
    F_vec = pair_forces_vec(pos, k=15.0, r0=1.5, rc=1.5)
    F_cel = pair_forces_cell(pos, L=10.0, k=15.0, r0=1.5, rc=1.5)
    dF_canon = float(np.max(np.abs(F_vec - F_cel)))

    p_v, v_v = pos.copy(), vel.copy()
    p_c, v_c = pos.copy(), vel.copy()
    Fp_v, Fp_c = None, None
    for _ in range(200):
        p_v, v_v, Fp_v = step_verlet(p_v, v_v, mass, dt=0.005, L=10.0, k=15.0, r0=1.5, rc=1.5, F_prev=Fp_v)
        p_c, v_c, Fp_c = step_verlet(p_c, v_c, mass, dt=0.005, L=10.0, k=15.0, r0=1.5, rc=1.5, F_prev=Fp_c,
                                     pair_force_fn=lambda p, k=15.0, r0=1.5, rc=1.5: pair_forces_cell(p, L=10.0, k=k, r0=r0, rc=rc))
    dpos_canon = float(np.max(np.abs(p_v - p_c)))
    print(f"Canonical (k=15.0, r0=1.5, rc=1.5, N=12): Force max|dF| = {dF_canon:.2e} N | 200-step max|dr| = {dpos_canon:.2e} m")
    rows.append(["canonical_k15_r0_1.5", 12, 10.0, 15.0, 1.5, 1.5, dF_canon, dpos_canon])

    # 2. Billiards Stiff Configuration (k=4000.0, r0=0.70, rc=0.70, N=16, L=10.0)
    pos_b, vel_b, mass_b = init_state(seed=42, N=16, L=10.0, dtype=np.float64)
    F_vec_b = pair_forces_vec(pos_b, k=4000.0, r0=0.70, rc=0.70)
    F_cel_b = pair_forces_cell(pos_b, L=10.0, k=4000.0, r0=0.70, rc=0.70)
    dF_bill = float(np.max(np.abs(F_vec_b - F_cel_b)))

    p_v, v_v = pos_b.copy(), vel_b.copy()
    p_c, v_c = pos_b.copy(), vel_b.copy()
    Fp_v, Fp_c = None, None
    for _ in range(200):
        p_v, v_v, Fp_v = step_verlet(p_v, v_v, mass_b, dt=0.0005, L=10.0, k=4000.0, r0=0.70, rc=0.70, F_prev=Fp_v)
        p_c, v_c, Fp_c = step_verlet(p_c, v_c, mass_b, dt=0.0005, L=10.0, k=4000.0, r0=0.70, rc=0.70, F_prev=Fp_c,
                                     pair_force_fn=lambda p, k=4000.0, r0=0.70, rc=0.70: pair_forces_cell(p, L=10.0, k=k, r0=r0, rc=rc))
    dpos_bill = float(np.max(np.abs(p_v - p_c)))
    print(f"Billiards (k=4000, r0=0.7, rc=0.7, N=16): Force max|dF| = {dF_bill:.2e} N | 200-step max|dr| = {dpos_bill:.2e} m")
    rows.append(["billiards_k4000_r0_0.7", 16, 10.0, 4000.0, 0.7, 0.7, dF_bill, dpos_bill])

    save_csv("task8_simulator_validation.csv",
             ["configuration", "N", "L", "k", "r0", "rc", "max_abs_force_diff", "step200_max_pos_diff"],
             rows, out_dir=out_dir)
    return rows


# Generate publication-standard Matplotlib figures
def make_plots(T, sim, out_dir=OUT):
    N = T[:, 0]
    t_loop, t_vec, t_cell, m_vec, m_cell = T[:, 1], T[:, 2], T[:, 3], T[:, 4], T[:, 5]

    def slope(x, y, lo):
        m = (x >= lo) & np.isfinite(y)
        if np.sum(m) < 2:
            return 1.0
        return float(np.polyfit(np.log(x[m]), np.log(y[m]), 1)[0])

    p_vec = slope(N, t_vec, 200)
    p_cell = slope(N, t_cell, 500)
    print(f"\nEmpirical scaling exponents (t ~ N^p): Vectorized O(N^2) p = {p_vec:.2f} | Cell list p = {p_cell:.2f}")

    # Figure 1: Runtime and speedup
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    m_lp = np.isfinite(t_loop)
    if np.any(m_lp):
        ax[0].loglog(N[m_lp], t_loop[m_lp], "s-", color="#7f8c8d", lw=1.2, label="Scalar Python Loop")
    m_vc = np.isfinite(t_vec)
    if np.any(m_vc):
        ax[0].loglog(N[m_vc], t_vec[m_vc], "o-", color="#c0392b", lw=1.4, label=f"NumPy Vectorized O(N$^{{2}}$) (slope {p_vec:.2f})")
    ax[0].loglog(N, t_cell, "^-", color="#2980b9", lw=1.6, label=f"Spatial Cell List (slope {p_cell:.2f})")

    idx_ref = 2 if len(N) > 2 else 0
    ref = t_cell[idx_ref] * (N / N[idx_ref])
    ax[0].loglog(N, ref, "k:", lw=1.0, label=r"Ideal $\mathcal{O}(N)$ Reference")
    ax[0].set_xlabel("Particle Count $N$ (Constant Density)")
    ax[0].set_ylabel("Execution Time per Force Call (s)")
    ax[0].set_title("Runtime Scaling Comparison")
    ax[0].legend(fontsize=8, frameon=True)
    ax[0].grid(True, which="both", alpha=0.3)

    sp = t_vec / t_cell
    valid_sp = np.isfinite(sp)
    if np.any(valid_sp):
        ax[1].semilogx(N[valid_sp], sp[valid_sp], "o-", color="#27ae60", lw=1.5)
        ax[1].axhline(1.0, color="gray", lw=0.8, ls="--")
        for x, y in zip(N[valid_sp], sp[valid_sp]):
            ax[1].annotate(f"{y:.1f}x", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    ax[1].set_xlabel("Particle Count $N$")
    ax[1].set_ylabel("Measured Speedup ($t_{\\mathrm{vec}} / t_{\\mathrm{cell}}$)")
    ax[1].set_title("Cell List Acceleration Factor")
    ax[1].grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(out_dir) / "task8_fig1_time_speedup.png", dpi=150)
    plt.close()

    # Figure 2: Peak memory scaling
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    Nth = np.logspace(np.log10(N[0]), np.log10(N[-1]), 50)
    ax.loglog(Nth, Nth ** 2 * 2 * 8 / 1e6, "k--", lw=1.0, label=r"Theoretical $N^2 \times 2 \times 8\mathrm{B}$")
    if np.any(valid_sp):
        ax.loglog(N[valid_sp], m_vec[valid_sp] / 1e6, "o-", color="#c0392b", lw=1.4, label="All-Pairs Vectorized (Peak Allocated)")
    ax.loglog(N, m_cell / 1e6, "^-", color="#2980b9", lw=1.5, label="Spatial Cell List (Peak Allocated)")
    ax.axhline(8000, color="red", lw=0.8, ls=":", label="8 GB Workstation RAM Limit")
    ax.set_xlabel("Particle Count $N$")
    ax.set_ylabel("Heap Memory Footprint (MB)")
    ax.set_title("Memory Scaling: Quadratic vs Linear")
    ax.legend(fontsize=8, frameon=True)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "task8_fig2_memory.png", dpi=150)
    plt.close()

    # Figure 3: Energy drift comparison
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    t = np.arange(len(sim["Eb"])) * sim["dt"]
    ax.plot(t, (sim["Eb"] - sim["Eb"][0]) / sim["Eb"][0], color="#c0392b", lw=1.5, label="All-Pairs (Vectorized)")
    ax.plot(t, (sim["Ec"] - sim["Ec"][0]) / sim["Ec"][0], "--", color="#2980b9", lw=1.5, label="Cell List")
    ax.set_xlabel("Time $t$ (s)")
    ax.set_ylabel("Relative Energy Drift $(E(t) - E(0)) / E(0)$")
    ax.set_title(f"Energy Conservation Invariance ($N={sim['N']}, \\Delta t={sim['dt']}$)")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "task8_fig3_energy_same.png", dpi=150)
    plt.close()


# Visual schematic of the cell list partition
def make_cell_picture(out_dir=OUT):
    n = 150
    pos, _, _, L = make_particles(n, seed=7)
    pos = pos * (30.0 / L)
    L = 30.0
    cx, cy, order, start, count, ncell = build_cells(pos, L, rc=R_CUT)
    w = L / ncell

    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    for g in np.arange(ncell + 1) * w:
        ax.axhline(g, color="#bdc3c7", lw=0.6)
        ax.axvline(g, color="#bdc3c7", lw=0.6)

    ax.scatter(pos[:, 0], pos[:, 1], s=14, color="#3498db", label="Particles")
    p = 0
    bx = (cx[p] - 1) * w
    by = (cy[p] - 1) * w
    ax.add_patch(plt.Rectangle((bx, by), 3 * w, 3 * w, fill=True, alpha=0.18, color="#e67e22", label=r"Adjacent $3 \times 3$ Cell Stencil"))
    ax.add_patch(plt.Circle(pos[p], R_CUT, fill=False, color="#c0392b", lw=1.5, label=r"Cutoff Radius $r_c$"))
    ax.scatter(*pos[p], s=65, color="#c0392b", zorder=5)

    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect("equal")
    ax.set_xlabel("Spatial Coordinate $x$")
    ax.set_ylabel("Spatial Coordinate $y$")
    ax.set_title(f"Cell Partition: $w = {w:.2f} \\geq r_c = {R_CUT}$")
    ax.legend(fontsize=8, loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(Path(out_dir) / "task8_fig0_cell_diagram.png", dpi=150)
    plt.close()


# Primary callable interface for master runner
def run_task_8(fast=False, output_dir=None):
    target_out = Path(output_dir) if output_dir else OUT
    target_out.mkdir(parents=True, exist_ok=True)

    print("Task 8: Spatial Partitioning & Cell List Benchmark")
    print(f"Target results folder: {target_out}")
    print(f"Mode: {'Fast' if fast else 'Full'}")

    exp_correctness(out_dir=target_out)
    T = exp_timing(fast=fast, out_dir=target_out)
    if not fast:
        exp_breakdown()
    sim = exp_simulation(fast=fast)
    exp_candidates(out_dir=target_out)
    exp_simulator_verification(out_dir=target_out)
    make_plots(T, sim, out_dir=target_out)
    make_cell_picture(out_dir=target_out)

    print(f"\nTask 8 complete. All output artifacts generated in {target_out}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 8: Spatial Partitioning via Cell-Linked Lists")
    parser.add_argument("--fast", action="store_true", help="Run accelerated test suite with smaller particle bounds")
    args = parser.parse_args()
    run_task_8(fast=args.fast)
