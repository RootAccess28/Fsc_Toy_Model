"""
Molecular / Atomic Toy Model Simulation Engine
===============================================
Unified physics engine, live visual simulation with real-time energy tracking,
numerical verification test suite (Task 1), and integrators (Tasks 2, 3, 4).
Master command center and architectural reference for Tasks 5, 6, 7, 8, and 9.

Usage:
    # 1. Dual-Panel Interactive Live GUI Simulation (Particle view + Real-time Energy graphs)
    python3 simulator.py
    python3 simulator.py --preset dissipative
    python3 simulator.py --preset strong_damping
    python3 simulator.py --preset euler_drift

    # 2. Task 1 Verification Suite (Memory layout, symmetry, conservation, momentum)
    python3 simulator.py --task1
    python3 simulator.py --verify

    # 3. Task 3 Timestep Accuracy & Energy Conservation Suite (Saves plots in 'task_3 results/')
    python3 simulator.py --task3

    # 4. Task 4 Dissipation & Relaxation Suite (Saves plots in 'task_4 results/')
    python3 simulator.py --task4

    # 5. Automated Sequential Batch: All Tasks (1, 3, and 4)
    python3 simulator.py --all-tasks

    # 6. Headless Execution / Animation Export
    python3 simulator.py --headless --frames 100
    python3 simulator.py --save-anim simulation.gif --frames 200

    # 7. Custom Physics / Integrator Parameters
    python3 simulator.py --N 24 --gamma 0.1 --dt 0.005 --method verlet
    python3 simulator.py --radius 0.35 --trail-len 40 --dt 0.002 --headless --frames 50
    python3 simulator.py --radius 0.1 --trail-len 10 --dt 0.005 --headless --frames 50
"""

import sys
import os
import argparse
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

# Ensure 'simulator' module is present in sys.modules when invoked directly as __main__
# to prevent redundant reloading or circular imports when task3_4 imports from simulator
if __name__ == "__main__" and "simulator" not in sys.modules:
    sys.modules["simulator"] = sys.modules[__name__]

# =============================================================================
# CENTRALIZED DEFAULT SIMULATION CONFIGURATION
# =============================================================================
# Anyone opening this script can adjust these default parameters right here
# in one single place at the top of the file without digging through code.
#
# These defaults govern interactive GUI runs, headless simulations, and presets,
# and can be dynamically overridden via CLI flags (--radius, --trail-len, --dt,
# --N, --L, --gamma, --k, --r0, --rc, --method, --preset, etc.).
# =============================================================================

DEFAULT_CONFIG = {
    # Geometric and Visualization Parameters
    # SYNCED: For spherical particles of radius R, contact occurs at center-to-center distance d = 2*R.
    # Hence, the spring rest length r0 = 2 * radius (and conversely radius = r0 / 2).
    "radius": 0.35,           # Particle ball radius (diameter 2R = 0.70 matches spring rest length r0)
    "trail_len": 24,          # Length of motion trail / tracer history buffer
    "N": 16,                  # Number of particles (sweet spot for smooth 60 FPS live visualization)
    "L": 10.0,                # Box dimensions: [0, L] x [0, L]

    # Time Discretization & Numerical Integrator
    "dt": 0.0005,             # Simulation time step dt (seconds) - resolves stiff high-velocity contact smoothly
    "substeps": 8,            # Physics integration sub-steps per animation frame (fast, silky motion)
    "method": "verlet",       # Integrator: 'verlet' (Velocity Verlet) or 'euler' (Forward Euler)
    "gamma": 0.0,             # Viscous damping coefficient (0.0 = conservative)

    # Force Field & Potential Parameters (Harmonic Soft Repulsion)
    # Perfectly synced with visual ball: r0 = 2 * radius (spring engages exact moment visual balls touch)
    "k": 4000.0,              # Elastic bumper stiffness (scaled for crisp, elastic bounce on contact)
    "r0": 0.70,               # Equilibrium distance = 2 * radius (exact physical contact threshold)
    "rc": 0.70,               # Interaction cutoff radius = r0

    # Presets, Initialization & Execution Settings
    "preset": "conservative", # Default preset mode ('conservative', 'dissipative', 'strong_damping', 'euler_drift')
    "seed": 42,               # Random number generator seed for reproducible particle distribution
    "v_max": 10.0,            # Maximum initial particle speed component per axis (energetic & fast)
    "frames": 1000,           # Maximum animation frames to render
    "hard_core": False,       # Pure continuous physics (False) vs geometric overlap separation (True)
    "backend": "numpy",       # Physics force engine: 'numpy' (vectorized) or 'c' (compiled pairforce.c)
    "dynamic_squish": True,   # Dynamic elastic ball contraction during collisions (squish & rebound)
}

# Exposed top-level aliases for direct, convenient access
DEFAULT_RADIUS = DEFAULT_CONFIG["radius"]
DEFAULT_TRAIL_LEN = DEFAULT_CONFIG["trail_len"]
DEFAULT_DT = DEFAULT_CONFIG["dt"]
DEFAULT_SUBSTEPS = DEFAULT_CONFIG["substeps"]
DEFAULT_N = DEFAULT_CONFIG["N"]
DEFAULT_L = DEFAULT_CONFIG["L"]
DEFAULT_GAMMA = DEFAULT_CONFIG["gamma"]
DEFAULT_K = DEFAULT_CONFIG["k"]
DEFAULT_R0 = DEFAULT_CONFIG["r0"]
DEFAULT_RC = DEFAULT_CONFIG["rc"]
DEFAULT_METHOD = DEFAULT_CONFIG["method"]
DEFAULT_V_MAX = DEFAULT_CONFIG["v_max"]
DEFAULT_PRESET = DEFAULT_CONFIG["preset"]

# =============================================================================
# 1. CORE PHYSICS & TASK 1 VERIFICATION IMPORTS
# =============================================================================
from core.physics import (
    pairwise_distances,
    pair_forces,
    total_forces,
    walls_reflect,
    particle_collisions,
    kinetic_energy,
    potential_energy,
    step_euler,
    step_verlet,
    init_state,
    run_sim,
)
from core.c_kernel import get_c_pair_forces_fn
from Task1_Verification.task1_verification import verify_task1, run_task_1

# =============================================================================
# 4. LIVE VISUAL SIMULATION WITH REAL-TIME ENERGY TRACKING (TASK 2)
# =============================================================================

# Preset definitions specialize the physics regime (e.g. damping, integrator, timestep)
# while inheriting all geometric and execution defaults dynamically from DEFAULT_CONFIG.
PRESET_SPECIALIZATIONS = {
    "conservative": {
        "description": "Conservative dynamics (gamma = 0.0): Total mechanical energy strictly conserved.",
        "gamma": 0.0,
        "method": "verlet",
    },
    "dissipative": {
        "description": "Dissipative damping (gamma = 0.15): Viscous drag drains kinetic energy into steady state.",
        "gamma": 0.15,
        "dt": 0.01,
        "method": "verlet",
    },
    "strong_damping": {
        "description": "Heavy damping (gamma = 0.50): Rapid relaxation to arrested equilibrium state.",
        "gamma": 0.50,
        "dt": 0.005,
        "method": "verlet",
    },
    "euler_drift": {
        "description": "Euler secular heating demo (gamma = 0.0, dt = 0.02): Demonstrates Euler energy divergence.",
        "gamma": 0.0,
        "dt": 0.02,
        "method": "euler",
    },
}

def get_preset_config(preset_name="conservative"):
    """
    Construct a full configuration dictionary dynamically by starting from DEFAULT_CONFIG
    and applying the specific physics modifications for the given preset.
    """
    cfg = DEFAULT_CONFIG.copy()
    if preset_name in PRESET_SPECIALIZATIONS:
        cfg.update(PRESET_SPECIALIZATIONS[preset_name])
    return cfg

# PRESETS dictionary dynamically populated from DEFAULT_CONFIG and preset specializations
PRESETS = {
    name: {**DEFAULT_CONFIG, **spec}
    for name, spec in PRESET_SPECIALIZATIONS.items()
}


def run_live_simulation(
    preset=None,
    gamma=None,
    dt=None,
    N=None,
    L=None,
    radius=None,
    k=None,
    r0=None,
    rc=None,
    method=None,
    seed=None,
    trail_len=None,
    frames=None,
    save_anim=None,
    headless=False,
    hard_core=None,
    v_max=None,
    substeps=None,
    backend=None,
    dynamic_squish=None,
):
    """
    Launches side-by-side real-time interactive simulation:
      - Left panel: Particle motion in 2D box with color-coded particles, tracers, and wall bounce.
      - Right panel: Live energy curves tracking K(t), U(t), and E_tot(t) simultaneously!

    All parameters cleanly fall back to preset specializations or DEFAULT_CONFIG
    when omitted (None), and can be dynamically overridden either programmatically
    or via CLI flags.
    """
    # 1. Resolve configuration from preset and overrides
    if preset is None:
        preset = DEFAULT_CONFIG.get("preset", "conservative")

    p_cfg = get_preset_config(preset)

    # Dynamic resolution: explicit argument > preset specialization > DEFAULT_CONFIG
    gamma = float(gamma if gamma is not None else p_cfg.get("gamma", DEFAULT_CONFIG["gamma"]))
    dt = float(dt if dt is not None else p_cfg.get("dt", DEFAULT_CONFIG["dt"]))
    N = int(N if N is not None else p_cfg.get("N", DEFAULT_CONFIG["N"]))
    L = float(L if L is not None else p_cfg.get("L", DEFAULT_CONFIG["L"]))
    k = float(k if k is not None else p_cfg.get("k", DEFAULT_CONFIG["k"]))
    method = str(method if method is not None else p_cfg.get("method", DEFAULT_CONFIG["method"])).lower()
    seed = int(seed if seed is not None else p_cfg.get("seed", DEFAULT_CONFIG["seed"]))
    trail_len = int(trail_len if trail_len is not None else p_cfg.get("trail_len", DEFAULT_CONFIG["trail_len"]))
    frames = int(frames if frames is not None else p_cfg.get("frames", DEFAULT_CONFIG["frames"]))
    v_max = float(v_max if v_max is not None else p_cfg.get("v_max", DEFAULT_CONFIG.get("v_max", 10.0)))
    hard_core = bool(hard_core if hard_core is not None else p_cfg.get("hard_core", DEFAULT_CONFIG.get("hard_core", False)))
    substeps = int(substeps if substeps is not None else p_cfg.get("substeps", DEFAULT_CONFIG.get("substeps", 8)))
    backend = str(backend if backend is not None else p_cfg.get("backend", DEFAULT_CONFIG.get("backend", "numpy"))).lower()
    dynamic_squish = bool(dynamic_squish if dynamic_squish is not None else p_cfg.get("dynamic_squish", DEFAULT_CONFIG.get("dynamic_squish", True)))

    # Force calculation backend selection (NumPy vs Compiled C via pairforce.c)
    pair_force_fn = None
    if backend == "c":
        try:
            pair_force_fn = get_c_pair_forces_fn()
            backend_label = "COMPILED C (pairforce.c) [~30x Faster]"
        except Exception as e:
            print(f"[BACKEND WARNING] Could not load C kernel ({e}), falling back to NumPy.")
            backend = "numpy"
            backend_label = "NUMPY (vectorized)"
            pair_force_fn = None
    else:
        backend_label = "NUMPY (vectorized)"

    # Synchronize physical spring distance (r0, rc) and visual particle radius (R):
    # Two spherical balls of radius R make physical contact when center distance d = 2 * R.
    # Therefore, the Hookean spring repulsion must engage at r0 = 2 * radius (and conversely radius = r0 / 2).
    if radius is not None and r0 is None:
        radius = float(radius)
        r0 = 2.0 * radius
        rc = float(rc if rc is not None else r0)
    elif r0 is not None and radius is None:
        r0 = float(r0)
        radius = r0 / 2.0
        rc = float(rc if rc is not None else r0)
    elif radius is None and r0 is None:
        radius = float(p_cfg.get("radius", DEFAULT_CONFIG["radius"]))
        r0 = float(p_cfg.get("r0", 2.0 * radius))
        rc = float(rc if rc is not None else p_cfg.get("rc", r0))
    else:
        radius = float(radius)
        r0 = float(r0)
        rc = float(rc if rc is not None else r0)

    print("\n" + "=" * 80)
    print(f"STARTING LIVE SIMULATION [{preset.upper()}]")
    print("=" * 80)
    print(f"Parameters: N={N}, L={L}, dt={dt}, gamma={gamma}, k={k}, r0={r0}, rc={rc}")
    print(f"Integrator: {method.upper()} | Radius: {radius} | Trail Len: {trail_len} | Substeps: {substeps} | Seed: {seed}")
    print(f"Engine    : {backend_label} | Elastic Squish: {'ON (Contracts on Collision)' if dynamic_squish else 'OFF'}")
    print("Close the simulation window to finish.")
    print("=" * 80 + "\n")

    if headless:
        matplotlib.use("Agg")

    # 2. Initialize particles
    # Calculate boundary margin to place particles cleanly inside box walls
    margin = max(radius + 0.05, min(radius + 0.8, L * 0.45)) if L > 2 * radius else L * 0.1
    pos, vel, mass = init_state(seed=seed, N=N, L=L, v_max=v_max, margin=margin)

    # Initial energies
    K0 = kinetic_energy(vel, mass)
    U0 = potential_energy(pos, k=k, r0=r0, rc=rc)
    E0 = K0 + U0

    # 3. Setup Figure: Left (Box simulation) & Right (Live Energy graph)
    fig, (ax_sim, ax_energy) = plt.subplots(1, 2, figsize=(14, 6.5))
    fig.canvas.manager.set_window_title(f"Particle Simulation [{preset}] - {method.upper()}")

    # --- Left Panel: Particle Motion ---
    ax_sim.set_xlim(-0.25, L + 0.25)
    ax_sim.set_ylim(-0.25, L + 0.25)
    ax_sim.set_aspect("equal")
    ax_sim.set_title(f"Particle Motion (N={N}, γ={gamma}, {method.upper()})", fontsize=11, fontweight="bold")
    ax_sim.set_xlabel("X coordinate", fontsize=10)
    ax_sim.set_ylabel("Y coordinate", fontsize=10)
    ax_sim.grid(True, linestyle=":", alpha=0.35)

    # Draw box boundary
    ax_sim.plot([0, L, L, 0, 0], [0, 0, L, L, 0], "k-", lw=2.2, label="Box Walls")

    colors = plt.cm.tab20(np.linspace(0, 1, max(N, 20)))[:N]
    trails = [ax_sim.plot([], [], "-", color=colors[i], alpha=0.5, lw=1.6)[0] for i in range(N)]
    draw_radius = radius if radius > 0.0 else 0.06
    circles = [
        plt.Circle((pos[i, 0], pos[i, 1]), draw_radius, facecolor=colors[i], edgecolor="black", lw=1.2, zorder=5)
        for i in range(N)
    ]
    for c in circles:
        ax_sim.add_patch(c)

    # Active collision spring indicators: dynamic connecting line rendered between balls during contact (r <= r0)
    spring_lines = LineCollection([], colors="#e63946", linewidths=2.4, alpha=0.9, zorder=6)
    ax_sim.add_collection(spring_lines)

    # --- Right Panel: Live Energy Graph ---
    line_K, = ax_energy.plot([], [], color="#1f77b4", lw=1.8, label="Kinetic K(t)")
    line_U, = ax_energy.plot([], [], color="#ff7f0e", lw=1.8, label="Potential U(t)")
    line_E, = ax_energy.plot([], [], color="#2ca02c", lw=2.4, label="Total E_tot(t)")
    ax_energy.axhline(E0, color="#7f7f7f", linestyle="--", lw=1.2, alpha=0.7, label=f"E(0) = {E0:.2f}")

    ax_energy.set_title("Real-Time Energy Tracking: K(t), U(t), E_tot(t)", fontsize=11, fontweight="bold")
    ax_energy.set_xlabel("Time t (s)", fontsize=10)
    ax_energy.set_ylabel("Mechanical Energy", fontsize=10)
    ax_energy.set_xlim(0, max(5.0, dt * 100))
    ax_energy.set_ylim(0, max(E0 * 1.35, 20.0))
    ax_energy.grid(True, linestyle="--", alpha=0.4)
    ax_energy.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    hud_text = ax_energy.text(
        0.03, 0.95, "", transform=ax_energy.transAxes,
        verticalalignment="top", fontsize=9, fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#f8f9fa", edgecolor="#ced4da", alpha=0.92)
    )

    # Animation state buffers
    t_hist = [0.0]
    K_hist = [K0]
    U_hist = [U0]
    E_hist = [E0]
    pos_hist = [pos.copy()]
    F_prev = [None]
    t_curr = [0.0]

    def update(frame):
        # Perform substeps physics iterations per visual frame for snappy, real-time animation speed
        for _ in range(substeps):
            if method == "euler":
                p_new, v_new = step_euler(pos, vel, mass, dt, L=L, gamma=gamma, k=k, r0=r0, rc=rc, radius=radius)
            else:
                p_new, v_new, F_new = step_verlet(
                    pos, vel, mass, dt, L=L, gamma=gamma, k=k, r0=r0, rc=rc, radius=radius,
                    F_prev=F_prev[0], pair_force_fn=pair_force_fn
                )
                F_prev[0] = F_new

            # Pure Physics Mode: repulsive spring forces handle 100% of the contact physics
            # continuously without artificial position-teleportation jumps.
            # Hard-core separation is only invoked if explicitly requested via --hard-core.
            if hard_core and radius > 0.0:
                p_new, v_new = particle_collisions(p_new, v_new, radius)

            pos[:] = p_new
            vel[:] = v_new
            t_curr[0] += dt

        # Measure energies
        K = kinetic_energy(vel, mass)
        U = potential_energy(pos, k=k, r0=r0, rc=rc)
        E = K + U

        t_hist.append(t_curr[0])
        K_hist.append(K)
        U_hist.append(U)
        E_hist.append(E)

        # Update particle positions and tracers dynamically based on trail_len
        if trail_len > 0:
            pos_hist.append(pos.copy())
            while len(pos_hist) > trail_len:
                pos_hist.pop(0)
            pts = np.array(pos_hist)
            for i in range(N):
                trails[i].set_data(pts[:, i, 0], pts[:, i, 1])
        else:
            for i in range(N):
                trails[i].set_data([], [])

        # Active collision spring indicators & Dynamic elastic ball contraction (squish & rebound)
        _, dist_mat, neigh_mat = pairwise_distances(pos, rc=r0)
        contact_pairs = []
        for i in range(N):
            for j in range(i + 1, N):
                if neigh_mat[i, j]:
                    contact_pairs.append([pos[i], pos[j]])
        spring_lines.set_segments(contact_pairs)

        if dynamic_squish:
            # Inter-particle compressive deformation (r0 - dist)
            pair_comp = np.where(neigh_mat, r0 - dist_mat, 0.0)
            max_pair_comp = np.max(pair_comp, axis=1) if N > 1 else np.zeros(N)

            # Wall collision compressive deformation
            wall_comp_x = np.maximum(0.0, np.maximum(draw_radius - pos[:, 0], pos[:, 0] - (L - draw_radius)))
            wall_comp_y = np.maximum(0.0, np.maximum(draw_radius - pos[:, 1], pos[:, 1] - (L - draw_radius)))
            max_wall_comp = np.maximum(wall_comp_x, wall_comp_y)

            total_comp = np.maximum(max_pair_comp, max_wall_comp)
            alpha_squish = 0.35
            r_min = 0.65 * draw_radius
            dyn_radii = np.maximum(r_min, draw_radius - alpha_squish * total_comp)

            for i in range(N):
                circles[i].center = (pos[i, 0], pos[i, 1])
                circles[i].set_radius(dyn_radii[i])
        else:
            for i in range(N):
                circles[i].center = (pos[i, 0], pos[i, 1])

        # Update energy curves
        line_K.set_data(t_hist, K_hist)
        line_U.set_data(t_hist, U_hist)
        line_E.set_data(t_hist, E_hist)

        # Dynamic axis scaling for energy panel
        cur_t = t_curr[0]
        if cur_t >= ax_energy.get_xlim()[1] * 0.95:
            ax_energy.set_xlim(0, max(5.0, cur_t * 1.4))

        all_e = K_hist + U_hist + E_hist
        max_e = max(all_e)
        min_e = min(all_e)
        curr_y_min, curr_y_max = ax_energy.get_ylim()
        if max_e > curr_y_max * 0.92 or min_e < curr_y_min:
            ax_energy.set_ylim(min(0.0, min_e - 2.0), max(curr_y_max, max_e * 1.25 + 5.0))

        # Update HUD text
        rel_err = abs(E - E0) / E0 if E0 > 0 else 0.0
        hud_text.set_text(
            f"t       = {cur_t:6.2f} s\n"
            f"K(t)    = {K:6.2f} J\n"
            f"U(t)    = {U:6.2f} J\n"
            f"E_tot   = {E:6.2f} J\n"
            f"|ΔE|/E0 = {rel_err:.2e}"
        )

        return circles + trails + [spring_lines, line_K, line_U, line_E, hud_text]

    if headless and not save_anim:
        print(f"Running headless simulation for {frames} frames...")
        for f in range(frames):
            update(f)
        print(f"Headless run completed {frames} steps. Final E_tot = {E_hist[-1]:.4f} J (K = {K_hist[-1]:.4f} J, U = {U_hist[-1]:.4f} J).")
        return None

    anim = FuncAnimation(fig, update, frames=frames, interval=25, blit=False)
    plt.tight_layout()

    if save_anim:
        out_path = Path(save_anim).resolve()
        print(f"Saving animation to {out_path} ...")
        if out_path.suffix.lower() == ".gif":
            anim.save(str(out_path), writer="pillow", fps=30)
        else:
            anim.save(str(out_path), fps=30)
        print("Animation saved successfully.")
    else:
        plt.show()

    return anim


# =============================================================================
# 5. TASK COMMAND DISPATCHERS (TASKS 1 - 7)
# =============================================================================

def run_task1():
    """Execute Task 1 numerical verification suite."""
    from Task1_Verification.task1_verification import run_task_1
    return run_task_1(headless=True)


def run_task3():
    """Execute Task 3 numerical experiments: Timestep accuracy & energy drift."""
    from Task3_Timestep_Accuracy.task3_accuracy import execute_task_3
    return execute_task_3()


def run_task4():
    """Execute Task 4 numerical experiments: Viscous dissipation & equilibrium relaxation."""
    from Task4_Viscous_Dissipation.task4_dissipation import execute_task_4
    return execute_task_4()


def run_task5():
    """Execute Task 5: float32 vs float64 precision study."""
    from Task5_Numerical_Precision.task5_precision_2d import run_task_5
    return run_task_5()


def run_task6():
    """Execute Task 6: Pair-force computational efficiency benchmark (Python vs NumPy vs C)."""
    from Task6_Computational_Efficiency.task6_benchmark import run_task_6
    return run_task_6()


def run_task7():
    """Execute Task 7: 2D & 3D displacement tensor memory and OOM scalability analysis."""
    from Task7_Memory_Scalability.task7_memory import run_task_7
    return run_task_7()


def run_all_tasks():
    """
    Execute Tasks 1, 3, 4, 5, 6, and 7 in an automated sequential batch.
    """
    print("=" * 86)
    print("UNIFIED COMMAND CENTER: RUNNING ALL EXPERIMENTAL & VERIFICATION TASKS (1, 3, 4, 5, 6, 7)")
    print("=" * 86)

    print("\n>>> [STEP 1/6] EXECUTING TASK 1: NUMERICAL VERIFICATION SUITE\n")
    success1 = run_task1()
    if not success1:
        print("\n[ABORT] Task 1 verification suite failed! Stopping sequence.")
        sys.exit(1)

    print("\n>>> [STEP 2/6] EXECUTING TASK 3: TIMESTEP ACCURACY & ENERGY CONSERVATION\n")
    run_task3()

    print("\n>>> [STEP 3/6] EXECUTING TASK 4: VISCOUS DISSIPATION & EQUILIBRIUM RELAXATION\n")
    run_task4()

    print("\n>>> [STEP 4/6] EXECUTING TASK 5: NUMERICAL PRECISION (FLOAT32 VS FLOAT64)\n")
    run_task5()

    print("\n>>> [STEP 5/6] EXECUTING TASK 6: EFFICIENCY BENCHMARK (PYTHON VS NUMPY VS C)\n")
    run_task6()

    print("\n>>> [STEP 6/6] EXECUTING TASK 7: DISPLACEMENT TENSOR MEMORY SCALING & OOM ANALYSIS\n")
    run_task7()

    print("\n" + "=" * 86)
    print("ALL EXPERIMENTAL TASKS (1, 3, 4, 5, 6, 7) COMPLETED SUCCESSFULLY!")
    print("=" * 86)


# =============================================================================
# 6. ARCHITECTURAL OVERVIEW & TASK MODULE ROADMAP
# =============================================================================
"""
Architecture Summary:
- Unified Physics Core (task3_4.py):
    Shared vectorised functions for pairwise distances, Hookean pair forces,
    symplectic Velocity Verlet integration, kinetic/potential energy calculations,
    and elastic wall reflections. All other task scripts import from here.
- Task 5 (Task5_2D_Report_and_Results/task5_precision_2d.py):
    Precision study (float32 vs float64) measuring energy drift and Lyapunov trajectory divergence.
- Task 6 (Task6_Revised_2D/task6_benchmark.py):
    Computational benchmarking comparing pure Python double loop, vectorized NumPy,
    and the compiled C kernel (pairforce.c via ctypes, achieving >70x speedup).
- Task 7 (fosc task 7 CODE.py):
    Displacement tensor memory scaling (2D vs 3D) and analytical Out-Of-Memory limit estimation.
- Interactive GUI & Engine (simulator.py):
    Live multi-panel particle animation with dynamic elastic ball squish on collision,
    optional compiled C backend acceleration (--backend c), and modular task dispatching.
"""


# =============================================================================
# 7. COMMAND LINE ENTRY POINT & MAIN ROUTER
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified Molecular/Atomic Physics Simulator and Verification Engine."
    )
    # Verification & Batch Experiment Flags
    parser.add_argument(
        "--task1", action="store_true",
        help="Run Task 1 numerical verification tests (memory layout, strides, symmetry, conservation, momentum)."
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Alias for --task1 (Run Task 1 numerical verification suite)."
    )
    parser.add_argument(
        "--task3", action="store_true",
        help="Run Task 3 numerical experiments (dt scan, energy conservation, convergence, save plots in 'task_3 results/')."
    )
    parser.add_argument(
        "--task4", action="store_true",
        help="Run Task 4 numerical experiments (viscous damping scan, relaxation times, save plots in 'task_4 results/')."
    )
    parser.add_argument(
        "--task5", action="store_true",
        help="Run Task 5 numerical precision study (float32 vs float64, energy drift, trajectory divergence)."
    )
    parser.add_argument(
        "--task6", action="store_true",
        help="Run Task 6 efficiency benchmark (Python double loop vs NumPy vectorized vs compiled C)."
    )
    parser.add_argument(
        "--task7", action="store_true",
        help="Run Task 7 2D & 3D displacement tensor memory scaling and OOM analysis."
    )
    parser.add_argument(
        "--all-tasks", action="store_true",
        help="Run Tasks 1, 3, 4, 5, 6, and 7 in sequential automated batch."
    )
    parser.add_argument(
        "--backend", type=str, choices=["numpy", "c"], default=None,
        help="Force calculation backend: 'numpy' (vectorized broadcasting) or 'c' (compiled ctypes kernel, pairforce.c)."
    )
    parser.add_argument(
        "--c-backend", dest="backend", action="store_const", const="c",
        help="Shortcut to enable the compiled C backend (pairforce.c via ctypes)."
    )
    parser.add_argument(
        "--dynamic-squish", dest="dynamic_squish", action="store_true", default=None,
        help="Enable dynamic visual ball contraction during collisions (elastic squish)."
    )
    parser.add_argument(
        "--no-squish", dest="dynamic_squish", action="store_false",
        help="Disable dynamic visual ball contraction (keep constant radius circles)."
    )

    # Interactive Simulation & Dynamic Physics Configuration Flags
    parser.add_argument(
        "--preset", type=str, default=None,
        choices=list(PRESETS.keys()),
        help=f"Simulation preset: {', '.join(PRESETS.keys())} (default: {DEFAULT_CONFIG['preset']})."
    )
    parser.add_argument("--N", type=int, default=None, help=f"Number of particles (default: {DEFAULT_CONFIG['N']}).")
    parser.add_argument("--L", type=float, default=None, help=f"Box side length (default: {DEFAULT_CONFIG['L']}).")
    parser.add_argument("--dt", type=float, default=None, help=f"Time step (default: {DEFAULT_CONFIG['dt']}).")
    parser.add_argument("--gamma", type=float, default=None, help=f"Viscous damping coefficient (default: {DEFAULT_CONFIG['gamma']}).")
    parser.add_argument("--k", type=float, default=None, help=f"Spring stiffness (default: {DEFAULT_CONFIG['k']}).")
    parser.add_argument("--r0", type=float, default=None, help=f"Equilibrium distance (default: {DEFAULT_CONFIG['r0']}).")
    parser.add_argument("--rc", type=float, default=None, help=f"Interaction cutoff radius (default: {DEFAULT_CONFIG['rc']}).")
    parser.add_argument("--radius", type=float, default=None, help=f"Particle radius (default: {DEFAULT_CONFIG['radius']}).")
    parser.add_argument("--trail-len", type=int, default=None, help=f"Length of particle motion tracers (default: {DEFAULT_CONFIG['trail_len']}).")
    parser.add_argument("--method", type=str, choices=["verlet", "euler"], default=None, help=f"Time integrator (default: {DEFAULT_CONFIG['method']}).")
    parser.add_argument("--seed", type=int, default=None, help=f"Random seed (default: {DEFAULT_CONFIG['seed']}).")
    parser.add_argument("--frames", type=int, default=None, help=f"Maximum number of animation frames (default: {DEFAULT_CONFIG['frames']}).")
    parser.add_argument("--v-max", "--velocity", dest="v_max", type=float, default=None, help=f"Maximum initial velocity magnitude (default: {DEFAULT_CONFIG.get('v_max', 14.0)}).")
    parser.add_argument("--substeps", type=int, default=None, help=f"Physics calculation sub-steps per animation frame (default: {DEFAULT_CONFIG.get('substeps', 8)}).")
    parser.add_argument("--save-anim", type=str, default=None, help="File path to save animation (e.g. anim.gif or anim.mp4).")
    parser.add_argument("--headless", action="store_true", help="Run without opening GUI window.")
    parser.add_argument(
        "--hard-core", dest="hard_core", action="store_true", default=None,
        help="Enable artificial geometric overlap separation (warning: causes discrete drops in potential energy)."
    )
    parser.add_argument(
        "--no-hard-core", dest="hard_core", action="store_false",
        help="Disable artificial geometric overlap separation."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Route 1: Run all tasks in batch
    if args.all_tasks:
        run_all_tasks()
        sys.exit(0)

    # Route 2: Run Task 1 verification suite
    if args.task1 or args.verify:
        success = run_task1()
        sys.exit(0 if success else 1)

    # Route 3: Run Task 3 experiments
    if args.task3:
        run_task3()
        sys.exit(0)

    # Route 4: Run Task 4 experiments
    if args.task4:
        run_task4()
        sys.exit(0)

    # Route 5: Run Task 5 experiments
    if args.task5:
        run_task5()
        sys.exit(0)

    # Route 6: Run Task 6 benchmark
    if args.task6:
        run_task6()
        sys.exit(0)

    # Route 7: Run Task 7 memory scaling analysis
    if args.task7:
        run_task7()
        sys.exit(0)

    # Route 8: Interactive Live Simulation (GUI or Headless)
    run_live_simulation(
        preset=args.preset,
        gamma=args.gamma,
        dt=args.dt,
        N=args.N,
        L=args.L,
        radius=args.radius,
        k=args.k,
        r0=args.r0,
        rc=args.rc,
        method=args.method,
        seed=args.seed,
        trail_len=args.trail_len,
        frames=args.frames,
        save_anim=args.save_anim,
        headless=args.headless,
        hard_core=args.hard_core,
        v_max=args.v_max,
        substeps=args.substeps,
        backend=args.backend,
        dynamic_squish=args.dynamic_squish,
    )


if __name__ == "__main__":
    main()
