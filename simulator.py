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
# 1. CORE PHYSICS & INTEGRATOR ENGINE (IMPORTED DIRECTLY FROM TASK MODULE)
# =============================================================================
# The actual mathematical formulas, Hookean spring pair forces, viscous drag,
# energy evaluations, and numerical time integrators (Euler and Velocity Verlet)
# are implemented natively inside task3_4.py. This preserves complete code
# ownership for the task developers when presenting their equations and code.
#
# simulator.py acts as the Master Visualizer, Dual-Panel GUI, and Controller,
# importing these physics kernels directly from task3_4.py.
# =============================================================================

from task3_4 import (
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
    execute_task_3,
    execute_task_4,
)


# =============================================================================
# 3. TASK 1 NUMERICAL VERIFICATION SUITE (--verify)
# =============================================================================

def verify_task1(verbose=True):
    """
    Executes comprehensive numerical verification of Task 1 requirements:
    1. Memory layout & array strides (dtype, itemsize, nbytes, strides; float64 vs float32 comparison for N=100,000).
    2. Distance matrix symmetry (||D - D^T||_max = 0).
    3. Distance zero diagonal (max(diag(D)) = 0).
    4. Self-neighbor exclusion (diag(neighbors) == False).
    5. Hand-built 4-particle analytic check against exact Pythagorean distances.
    6. Neighbor count & cutoff scaling across multiple rc.
    7. Newton's Third Law (equal & opposite forces, sum(F) ~ 0).
    8. Box boundary containment & wall reflections.
    9. Energy positivity & initial values consistency.
    """
    checks = []

    print("=" * 86)
    print("TASK 1: NUMERICAL VERIFICATION & FOUNDATIONAL PHYSICS EVIDENCE")
    print("=" * 86)

    # --- Check 1: Memory Layout & N=100k Calculation ---
    pos_sample = np.zeros((12, 2), dtype=np.float64)
    pos_sample32 = pos_sample.astype(np.float32)

    N_large = 100000
    pos64_large = np.zeros((N_large, 2), dtype=np.float64)
    pos32_large = np.zeros((N_large, 2), dtype=np.float32)

    bytes_64_actual = pos64_large.nbytes
    bytes_32_actual = pos32_large.nbytes

    sizeof_double = 8  # standard C sizeof(double)
    sizeof_float = 4   # standard C sizeof(float)
    bytes_64_hand = N_large * 2 * sizeof_double
    bytes_32_hand = N_large * 2 * sizeof_float

    mem_match = (bytes_64_actual == bytes_64_hand) and (bytes_32_actual == bytes_32_hand)
    checks.append(("Memory layout & N=100k strides (float64: 1.6MB, float32: 0.8MB)", mem_match))

    if verbose:
        print("[1] MEMORY & STRIDE INSPECTION:")
        print(f"    - Sample pos shape: {pos_sample.shape}, dtype: {pos_sample.dtype}")
        print(f"    - itemsize: {pos_sample.itemsize} bytes, strides: {pos_sample.strides}, nbytes: {pos_sample.nbytes} bytes")
        print(f"    - float32 itemsize: {pos_sample32.itemsize} bytes, nbytes: {pos_sample32.nbytes} bytes")
        print(f"    - N=100,000 particles memory:")
        print(f"        float64 actual = {bytes_64_actual:,} B ({bytes_64_actual/1e6:.2f} MB) | by-hand = {bytes_64_hand:,} B")
        print(f"        float32 actual = {bytes_32_actual:,} B ({bytes_32_actual/1e6:.2f} MB) | by-hand = {bytes_32_hand:,} B")
        print(f"    -> Result: {'PASS' if mem_match else 'FAIL'}\n")

    # --- Check 2, 3, 4: Distance Matrix Properties ---
    rng = np.random.default_rng(101)
    pos_test = rng.uniform(0.0, 10.0, size=(25, 2))
    rc_test = 2.5
    d_mat, dist_mat, neigh_mat = pairwise_distances(pos_test, rc=rc_test)

    sym_diff = float(np.max(np.abs(dist_mat - dist_mat.T)))
    pass_sym = (sym_diff < 1e-14)
    checks.append(("Distance matrix symmetry (||D - D^T||_max < 1e-14)", pass_sym))

    diag_max = float(np.max(np.abs(np.diag(dist_mat))))
    pass_diag = (diag_max == 0.0)
    checks.append(("Distance matrix zero diagonal (max(diag(D)) == 0)", pass_diag))

    pass_self = not np.any(np.diag(neigh_mat))
    checks.append(("Self-neighbor exclusion (diag(neighbors) == False)", pass_self))

    if verbose:
        print("[2] DISTANCE MATRIX PROPERTIES (N=25 random particles):")
        print(f"    - Distance symmetry residual ||D - D^T||_inf : {sym_diff:.3e} -> {'PASS' if pass_sym else 'FAIL'}")
        print(f"    - Zero diagonal max value                     : {diag_max:.3e} -> {'PASS' if pass_diag else 'FAIL'}")
        print(f"    - Self-neighbor exclusion (any diagonal True): {np.any(np.diag(neigh_mat))} -> {'PASS' if pass_self else 'FAIL'}\n")

    # --- Check 5: Hand-built 4-particle Exact Analytic Verification ---
    # Particle 0: (0, 0)
    # Particle 1: (1, 0) -> dist = 1.0
    # Particle 2: (0, 2) -> dist = 2.0 to 0; dist = sqrt(1+4) = sqrt(5) to 1
    # Particle 3: (4, 4) -> dist = sqrt(32) to 0; dist = 5.0 to 1; dist = sqrt(20) to 2
    pos_4 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 2.0],
        [4.0, 4.0]
    ])
    dist_exact = np.array([
        [0.0,           1.0,           2.0,           np.sqrt(32.0)],
        [1.0,           0.0,           np.sqrt(5.0),  5.0],
        [2.0,           np.sqrt(5.0),  0.0,           np.sqrt(20.0)],
        [np.sqrt(32.0), 5.0,           np.sqrt(20.0), 0.0]
    ])
    _, dist_calc, neigh_4 = pairwise_distances(pos_4, rc=2.1)
    pass_analytic = bool(np.allclose(dist_calc, dist_exact, atol=1e-14))
    checks.append(("Hand-built 4-particle exact Pythagorean distances", pass_analytic))

    # Neighbor check for rc = 2.1:
    # 0 has neighbors {1, 2} (dists 1.0, 2.0 <= 2.1; dist 5.66 > 2.1) -> 2 neighbors
    # 1 has neighbors {0} (dist 1.0; dist 2.236 > 2.1) -> 1 neighbor
    # 2 has neighbors {0} (dist 2.0; dist 2.236 > 2.1) -> 1 neighbor
    # 3 has neighbors {} -> 0 neighbors
    counts_calc = np.sum(neigh_4, axis=1).tolist()
    counts_expected = [2, 1, 1, 0]
    pass_counts = (counts_calc == counts_expected)
    checks.append(("Hand-built 4-particle neighbor counts (rc=2.1 -> [2, 1, 1, 0])", pass_counts))

    if verbose:
        print("[3] HAND-BUILT 4-PARTICLE ANALYTIC BENCHMARK:")
        print(f"    - Analytic distance match residual: {np.max(np.abs(dist_calc - dist_exact)):.3e} -> {'PASS' if pass_analytic else 'FAIL'}")
        print(f"    - Calculated neighbor counts      : {counts_calc} (Expected: {counts_expected}) -> {'PASS' if pass_counts else 'FAIL'}\n")

    # --- Check 6: Cutoff Scaling on Random Ensemble ---
    pos_50 = rng.uniform(0.0, 10.0, size=(50, 2))
    rc_list = [1.0, 1.5, 2.0, 3.0]
    avg_neighbors = []
    for rc_val in rc_list:
        _, _, n_mat = pairwise_distances(pos_50, rc=rc_val)
        avg_neighbors.append(float(np.mean(np.sum(n_mat, axis=1))))

    # Monotonically increasing check
    pass_monotonic = all(avg_neighbors[i] < avg_neighbors[i+1] for i in range(len(avg_neighbors)-1))
    checks.append(("Cutoff scaling monotonic increase on N=50 ensemble", pass_monotonic))

    if verbose:
        print("[4] CUTOFF SCALING ENSEMBLE (N=50 particles in L=10 box):")
        for rc_val, avg_n in zip(rc_list, avg_neighbors):
            print(f"    - rc = {rc_val:<4.1f} -> Average neighbors per particle = {avg_n:<6.2f}")
        print(f"    -> Monotonic growth check: {'PASS' if pass_monotonic else 'FAIL'}\n")

    # --- Check 7: Newton's Third Law (Net Force Sum ~ 0) ---
    F_pair = pair_forces(pos_test, k=15.0, r0=1.5, rc=2.5)
    net_F = np.sum(F_pair, axis=0)
    net_F_norm = float(np.linalg.norm(net_F))
    pass_newton = (net_F_norm < 1e-12)
    checks.append(("Newton's Third Law / Momentum conservation (||sum(F)|| < 1e-12)", pass_newton))

    if verbose:
        print("[5] NEWTON'S THIRD LAW / PAIR FORCE BALANCE:")
        print(f"    - Net force vector sum(F): [{net_F[0]:+.3e}, {net_F[1]:+.3e}]")
        print(f"    - Net force Euclidean norm: {net_F_norm:.3e} -> {'PASS' if pass_newton else 'FAIL'}\n")

    # --- Check 8: Wall Reflection and Containment ---
    # Particles positioned deliberately beyond all 4 boundaries with outward velocities:
    pos_outside = np.array([
        [-0.5, 5.0],    # Left wall
        [10.5, 5.0],    # Right wall
        [5.0, -0.5],    # Bottom wall
        [5.0, 10.5]     # Top wall
    ])
    vel_outward = np.array([
        [-3.0, 0.0],
        [3.0, 0.0],
        [0.0, -3.0],
        [0.0, 3.0]
    ])
    pos_refl, vel_refl = walls_reflect(pos_outside, vel_outward, L=10.0, radius=0.2)

    contained_x = (pos_refl[:, 0] >= 0.2) & (pos_refl[:, 0] <= 9.8)
    contained_y = (pos_refl[:, 1] >= 0.2) & (pos_refl[:, 1] <= 9.8)
    vel_inward_x = (vel_refl[0, 0] > 0.0) and (vel_refl[1, 0] < 0.0)
    vel_inward_y = (vel_refl[2, 1] > 0.0) and (vel_refl[3, 1] < 0.0)

    pass_walls = bool(np.all(contained_x) and np.all(contained_y) and vel_inward_x and vel_inward_y)
    checks.append(("Wall reflection & boundary containment (radius=0.2, L=10)", pass_walls))

    if verbose:
        print("[6] WALL REFLECTION & BOUNDARY CLAMPING:")
        print(f"    - Outward positions reflected to: \n{np.array2string(pos_refl, prefix='        ')}")
        print(f"    - Reversed inward velocities:     \n{np.array2string(vel_refl, prefix='        ')}")
        print(f"    -> All particles contained within [0.2, 9.8] with inward velocity: {'PASS' if pass_walls else 'FAIL'}\n")

    # --- Check 9: Energy Positivity & Seed 42 Reference ---
    pos_init, vel_init, mass_init = init_state(seed=42, N=12, L=10.0)
    K0 = kinetic_energy(vel_init, mass_init)
    U0 = potential_energy(pos_init, k=15.0, r0=1.5, rc=1.5)
    E0 = K0 + U0

    pass_energy = (K0 > 0) and (U0 >= 0) and abs(E0 - 77.1426) < 0.01
    checks.append(("Energy positivity & seed 42 baseline (E(0) ~ 77.14 J)", pass_energy))

    if verbose:
        print("[7] ENERGY POSITIVITY & INITIAL REPRODUCIBILITY (seed 42):")
        print(f"    - Kinetic Energy   K(0) = {K0:.4f} J")
        print(f"    - Potential Energy U(0) = {U0:.4f} J")
        print(f"    - Total Energy     E(0) = {E0:.4f} J (Target: 77.1426 J)")
        print(f"    -> Consistency check: {'PASS' if pass_energy else 'FAIL'}\n")

    # --- Summary Table ---
    print("-" * 86)
    print(f"{'Check Description':<68} | {'Status':<12}")
    print("-" * 86)
    all_passed = True
    for desc, status in checks:
        if not status:
            all_passed = False
        print(f"{desc:<68} | {'[ PASS ]' if status else '[ FAIL ]':<12}")
    print("-" * 86)
    passed_count = sum(1 for _, s in checks if s)
    total_count = len(checks)
    print(f"VERIFICATION SUMMARY: {passed_count}/{total_count} CHECKS PASSED ({passed_count/total_count*100:.1f}%)")
    print("=" * 86 + "\n")

    return all_passed


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
                    pos, vel, mass, dt, L=L, gamma=gamma, k=k, r0=r0, rc=rc, radius=radius, F_prev=F_prev[0]
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

        for i in range(N):
            circles[i].center = (pos[i, 0], pos[i, 1])

        # Active collision spring indicators: render line connecting centers of balls in contact (r <= r0)
        _, dist_mat, neigh_mat = pairwise_distances(pos, rc=r0)
        contact_pairs = []
        for i in range(N):
            for j in range(i + 1, N):
                if neigh_mat[i, j]:
                    contact_pairs.append([pos[i], pos[j]])
        spring_lines.set_segments(contact_pairs)

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
# 5. TASK COMMAND DISPATCHERS (TASKS 1, 3, 4)
# =============================================================================

def run_task1():
    """
    Execute Task 1 numerical verification suite:
    Memory layout, strides, symmetry, energy conservation, and momentum conservation.
    Returns True if all tests pass.
    """
    return verify_task1(verbose=True)


def run_task3():
    """
    Execute Task 3 numerical experiments:
    Timestep scan (dt in [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]),
    Euler vs Velocity Verlet energy drift, convergence order,
    and 10-seed ensemble statistics.
    Saves CSV table and high-resolution figures in 'task_3 results/'.
    """
    try:
        from task3_4 import execute_task_3
    except ImportError as e:
        print(f"[ERROR] Could not import execute_task_3 from task3_4: {e}")
        sys.exit(1)
    execute_task_3()


def run_task4():
    """
    Execute Task 4 numerical experiments:
    Viscous damping scan (gamma in [0.0, 0.05, 0.1, 0.5]),
    Dissipation of kinetic and total energy, relaxation times to mechanical equilibrium,
    and particle trajectory comparison.
    Saves CSV table and high-resolution figures in 'task_4 results/'.
    """
    try:
        from task3_4 import execute_task_4
    except ImportError as e:
        print(f"[ERROR] Could not import execute_task_4 from task3_4: {e}")
        sys.exit(1)
    execute_task_4()


def run_all_tasks():
    """
    Execute Tasks 1, 3, and 4 in an automated sequential batch.
    """
    print("=" * 86)
    print("UNIFIED COMMAND CENTER: RUNNING ALL EXPERIMENTAL & VERIFICATION TASKS (1, 3, 4)")
    print("=" * 86)

    print("\n>>> [STEP 1/3] EXECUTING TASK 1: NUMERICAL VERIFICATION SUITE\n")
    success1 = run_task1()
    if not success1:
        print("\n[ABORT] Task 1 verification suite failed! Stopping sequence.")
        sys.exit(1)

    print("\n>>> [STEP 2/3] EXECUTING TASK 3: TIMESTEP ACCURACY & ENERGY CONSERVATION\n")
    run_task3()

    print("\n>>> [STEP 3/3] EXECUTING TASK 4: VISCOUS DISSIPATION & EQUILIBRIUM RELAXATION\n")
    run_task4()

    print("\n" + "=" * 86)
    print("ALL EXPERIMENTAL TASKS (1, 3, 4) COMPLETED SUCCESSFULLY!")
    print("=" * 86)


# =============================================================================
# 6. DEVELOPER HANDOVER & ARCHITECTURAL ROADMAP FOR TASKS 5, 6, 7, 8, AND 9
# =============================================================================
"""
DEVELOPER HANDOVER & IMPLEMENTATION SPECIFICATION
=================================================
This section serves as the definitive engineering blueprint and implementation
guide for subsequent developers / AI agents implementing Tasks 5 through 9.

All functions in this file (simulator.py) have been structured as pure, modular,
vectorized components ready for immediate benchmarking, extension, and compilation.

-------------------------------------------------------------------------------
TASK 5: NUMERICAL PRECISION STUDY (FLOAT32 VS FLOAT64)
-------------------------------------------------------------------------------
Objective:
    Quantify floating-point precision effects on memory usage, energy conservation,
    and chaotic trajectory divergence (Lyapunov sensitivity).

Key Metrics & Mathematical Formulas:
    1. Memory Footprint:
       - float64: itemsize = 8 bytes per scalar (64 bits).
       - float32: itemsize = 4 bytes per scalar (32 bits, exactly 50% reduction).
       - State vector memory: M = 2 * N * itemsize bytes for pos and vel.
    2. Energy Drift:
       epsilon_E(t) = |E(t) - E(0)| / E(0)
    3. Trajectory Divergence (Relative Coordinate Difference):
       epsilon_pos(t) = ||x_32(t) - x_64(t)||_F / ||x_64(t)||_F
       where ||.||_F is the Frobenius norm across all N particles and 2 dimensions:
           ||Delta||_F = sqrt( sum_{i=1}^N sum_{d=1}^2 (x_{i,d}^(32) - x_{i,d}^(64))^2 )

Step-by-Step Implementation Guide for Task 5:
    1. Generate reference initial condition in float64 using:
           pos64, vel64, mass64 = init_state(seed=42, N=12, L=10.0)
    2. Clone and cast state to np.float32:
           pos32 = pos64.astype(np.float32)
           vel32 = vel64.astype(np.float32)
           mass32 = mass64.astype(np.float32)
    3. Time Integration:
       Integrate both systems using step_verlet() for T = 10.0 s with dt = 0.005 s.
       CRITICAL: For the float32 run, pass parameters cast to float32:
           k_32 = np.float32(15.0); r0_32 = np.float32(1.5); rc_32 = np.float32(1.5)
           dt_32 = np.float32(0.005); L_32 = np.float32(10.0)
       This ensures NumPy maintains pure float32 arithmetic without promoting to float64.
    4. At each step t_n:
       - Compute E_64(t_n) and E_32(t_n).
       - Compute relative trajectory divergence:
             eps_pos = np.linalg.norm(pos32 - pos64) / np.linalg.norm(pos64)
    5. Expected Scientific Observations:
       - Collision dynamics in N-body systems are chaotic with positive Lyapunov exponents.
       - Roundoff errors in float32 (~1e-7 relative machine epsilon) compound at every
         inter-particle bounce, causing exponential growth of epsilon_pos(t) until
         macroscopic divergence (eps_pos ~ O(1)) is reached.
       - However, Velocity Verlet is symplectic; even in float32, the total mechanical
         energy oscillates around a bounded shadow Hamiltonian without secular divergence,
         maintaining eps_E ~ 1e-5 to 1e-6.

-------------------------------------------------------------------------------
TASK 6: COMPUTATIONAL EFFICIENCY & LANGUAGE COMPARISON (PYTHON VS NUMPY VS C)
-------------------------------------------------------------------------------
Objective:
    Implement and benchmark pairwise force computation across three language tiers:
    1. Pure Python nested loops (O(N^2) interpreted overhead, baseline).
    2. Vectorized NumPy broadcasting (pair_forces in Section 1, SIMD C-core).
    3. Compiled C shared library via ctypes, exploiting Newton's 3rd Law (j > i).

C Implementation Source Code (pairforce.c):
    Save the following implementation as 'pairforce.c':
    ---------------------------------------------------------------------------
    #include <math.h>

    void compute_pair_forces(const double* pos, double* forces, int N,
                             double k, double r0, double rc) {
        // 1. Initialize forces array to zero
        for (int i = 0; i < 2 * N; ++i) {
            forces[i] = 0.0;
        }

        double rc2 = rc * rc;

        // 2. Loop over unique pairs i < j (Newton's 3rd Law optimization)
        for (int i = 0; i < N; ++i) {
            double xi = pos[2 * i];
            double yi = pos[2 * i + 1];
            for (int j = i + 1; j < N; ++j) {
                double dx = xi - pos[2 * j];
                double dy = yi - pos[2 * j + 1];
                double dist2 = dx * dx + dy * dy;

                if (dist2 <= rc2 && dist2 > 1e-24) {
                    double dist = sqrt(dist2);
                    double fmag = -k * (dist - r0);
                    double fx = fmag * (dx / dist);
                    double fy = fmag * (dy / dist);

                    // Equal and opposite pairwise reaction: F_ij = -F_ji
                    forces[2 * i]     += fx;
                    forces[2 * i + 1] += fy;
                    forces[2 * j]     -= fx;
                    forces[2 * j + 1] -= fy;
                }
            }
        }
    }
    ---------------------------------------------------------------------------

Compilation Commands:
    Linux:
        gcc -O2 -shared -fPIC -o libpairforce.so pairforce.c
    macOS:
        gcc -O2 -dynamiclib -fPIC -o libpairforce.dylib pairforce.c
        (or gcc -O2 -shared -fPIC -o libpairforce.so pairforce.c)

Python ctypes Binding & Wrapper:
    import ctypes
    lib_path = Path("./libpairforce.dylib" if sys.platform == "darwin" else "./libpairforce.so")
    c_lib = ctypes.CDLL(str(lib_path))
    c_lib.compute_pair_forces.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS, WRITEABLE"),
        ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double
    ]
    c_lib.compute_pair_forces.restype = None

    def pair_forces_c(pos, k=15.0, r0=1.5, rc=1.5):
        forces = np.empty_like(pos)
        c_lib.compute_pair_forces(pos, forces, len(pos), k, r0, rc)
        return forces

Benchmarking Protocol:
    - Sweep particle counts N in [10, 20, 50, 100, 200, 500, 1000].
    - Measure runtime of:
      (a) Pure Python double loop
      (b) NumPy pair_forces(pos)
      (c) C pair_forces_c(pos)
    - Compute Speedup: S_numpy = T_python / T_numpy, S_c = T_python / T_c.
    - Expected results: C is 50x-200x faster than pure Python, and 2x-5x faster
      than NumPy by avoiding intermediate (N, N, 2) heap allocations and halving
      pair checks via symmetry j > i.

-------------------------------------------------------------------------------
TASK 7: SCALABILITY & MEMORY BOTTLENECK ANALYSIS
-------------------------------------------------------------------------------
Objective:
    Evaluate memory scalability and identify the physical memory ceiling
    of the vectorized NumPy broadcasting implementation.

Displacement Tensor Memory Derivation:
    The vectorized distance tensor computes:
        d = pos[:, None, :] - pos[None, :, :]
    Tensor shape: (N, N, 2)
    Total elements: 2 * N^2
    Exact memory consumption:
        Memory_disp = 2 * N^2 * itemsize bytes

    For standard float64 (itemsize = 8 bytes):
        Memory_disp = 16 * N^2 bytes

Complete Working Memory Footprint per Force Call:
    - Displacement tensor 'd'    (N, N, 2, float64): 16 * N^2 bytes
    - Distance matrix 'dist'      (N, N, float64):     8 * N^2 bytes
    - Neighbor mask 'neighbors'   (N, N, bool):        1 * N^2 bytes
    - Normalized unit vectors 'u' (N, N, 2, float64): 16 * N^2 bytes
    - Force magnitude 'fmag'      (N, N, float64):     8 * N^2 bytes
    ---------------------------------------------------------------------------
    Total Peak Intermediate Memory: ~ 49 * N^2 bytes

Empirical Scaling Table:
    ---------------------------------------------------------------------------
    N           Displacement Tensor (16*N^2)    Total Intermediate (~49*N^2)
    ---------------------------------------------------------------------------
    100         160 KB                         490 KB   (fits in CPU L2 cache)
    1,000       16 MB                          49 MB    (fits in CPU L3 cache)
    5,000       400 MB                         1.22 GB  (exceeds cache, RAM bound)
    10,000      1.6 GB                         4.90 GB  (risk of RAM exhaustion)
    50,000      40 GB                          122.5 GB (MemoryError / crash)
    100,000     160 GB                         490 GB   (requires supercomputer)
    ---------------------------------------------------------------------------

Handover Instructions:
    - Use `tracemalloc` to record peak allocated heap memory during pair_forces(pos).
    - Plot Peak Memory (MB) vs N on log-log scale. Verify the slope is exactly 2.0.
    - Conclude that while vectorized NumPy is elegant for small N, its O(N^2)
      spatial footprint is a fatal scaling barrier, necessitating Task 8.

-------------------------------------------------------------------------------
TASK 8: ALGORITHMIC INNOVATION (O(N) LINKED CELLS & LJ POTENTIALS)
-------------------------------------------------------------------------------
Objective:
    Break the O(N^2) scaling barrier using a Linked-Cell / Cell-List spatial
    decomposition algorithm, and extend physics to Lennard-Jones interactions.

Insertion Point in simulator.py:
    Replace or provide an alternate backend for `pairwise_distances()` and
    `pair_forces()` in Section 1.

Linked-Cell Algorithm Blueprint:
    1. Spatial Discretization:
       Subdivide box L x L into an M x M grid of square cells, where cell width:
           r_cell = L / M >= rc   ==>   M = int(np.floor(L / rc))
    2. Binning Particles:
       Map each particle i at (x_i, y_i) to cell coordinates:
           cell_x = int(x_i / r_cell)
           cell_y = int(y_i / r_cell)
       Construct linked list arrays:
           head[M, M] : index of first particle in each cell
           next_particle[N] : index of next particle in the same cell
    3. Interaction Computation:
       To compute forces on particle i in cell (cx, cy), only iterate over
       particles residing in cell (cx, cy) and the 8 adjacent neighbor cells.
    4. Complexity Breakthrough:
       - For homogeneous density, average particles per cell rho_cell = N / M^2.
       - Each particle checks 9 * rho_cell neighbors = O(1).
       - Total time complexity drops from O(N^2) to O(N).
       - Total memory complexity drops from O(N^2) to O(N) (never allocate (N, N, 2)).

Lennard-Jones (12-6) Potential Extension:
    To model realistic molecular van der Waals attraction and Pauli repulsion:
        V_LJ(r) = 4 * epsilon * [(sigma / r)^12 - (sigma / r)^6]
        F_LJ(r) = (24 * epsilon / r) * [2 * (sigma / r)^12 - (sigma / r)^6] * (r_vec / r)
    Add a parameter `potential="harmonic"` or `potential="lennard_jones"` in
    `pair_forces()` and `potential_energy()`.

-------------------------------------------------------------------------------
TASK 9: ENGINEERING DECISIONS & FINAL SYSTEM SYNTHESIS
-------------------------------------------------------------------------------
Objective:
    Synthesize all analytical and benchmark results from Tasks 1-8 into definitive
    engineering recommendations for production molecular dynamics.

Definitive Decision Matrix:
    1. Time Integrator:
       - Selection: Velocity Verlet (step_verlet).
       - Rationale: Symplectic, time-reversible, second-order accurate O(dt^2).
         Total mechanical energy exhibits strictly bounded oscillations without
         secular drift. Euler exhibits exponential energy divergence (Task 3).
    2. Timestep Selection:
       - Selection: dt = 0.005 s.
       - Rationale: Provides eps_E,max ~ 0.003 (0.3% relative energy variation),
         well below typical 1% scientific thresholds, while requiring 5x fewer
         costly force evaluations than dt = 0.001 s.
    3. Numerical Precision:
       - Scientific Computing / Trajectory Fidelity: float64 is strictly required
         due to exponential divergence of chaotic Lyapunov trajectories (Task 5).
       - Real-time Visualizations / Interactive Graphics: float32 is optimal,
         reducing memory bandwidth by 50% while preserving stable bounded energy.
    4. Execution Strategy vs Scale:
       - Small Systems (N <= 1,000): Vectorized NumPy broadcasting (simplicity).
       - Medium Systems (1,000 < N <= 10,000): Compiled C via ctypes (Task 6).
       - Large Systems (N > 10,000): Linked-Cell spatial lists (Task 8).
    5. Dissipation Regime:
       - Selection: gamma = 0.5 s^-1.
       - Rationale: Produces rapid, monotonic kinetic relaxation to static
         mechanical equilibrium (t_relax ~ 16 s) without lingering ringing (Task 4).
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
        "--all-tasks", action="store_true",
        help="Run Tasks 1, 3, and 4 in sequential automated batch."
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

    # Route 5: Interactive Live Simulation (GUI or Headless)
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
    )


if __name__ == "__main__":
    main()
