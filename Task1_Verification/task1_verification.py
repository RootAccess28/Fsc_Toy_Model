"""
Task 1: Numerical Verification Suite
====================================
Comprehensive numerical verification proving:
1. Memory layout & array strides (float64 vs float32 comparison for N=100,000).
2. Distance matrix symmetry (||D - D^T||_max = 0) and zero diagonal.
3. Self-neighbor exclusion.
4. Hand-built 4-particle analytic check against exact Pythagorean distances.
5. Neighbor count & cutoff scaling monotonic check across multiple rc.
6. Newton's Third Law (equal and opposite forces, sum(F) ~ 0).
7. Box boundary containment & specular wall reflections.
8. Energy positivity & initial values consistency.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
from core.physics import (
    pairwise_distances,
    pair_forces,
    walls_reflect,
    init_state,
    kinetic_energy,
    potential_energy,
)


def verify_task1(verbose=True):
    """
    Executes the 10-check verification suite and prints formatted evidence.
    Returns True if all 10 checks pass, False otherwise.
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


def run_task_1(headless=True):
    return verify_task1(verbose=True)


if __name__ == "__main__":
    success = run_task_1()
    sys.exit(0 if success else 1)
