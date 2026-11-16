"""Task 1: Numerical verification suite."""

import sys
from pathlib import Path

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
    checks = []
    print("Task 1: Numerical Verification Suite")
    print("-" * 50)

    # 1. Memory layout and N=100k calculation
    pos_sample = np.zeros((12, 2), dtype=np.float64)
    pos_sample32 = pos_sample.astype(np.float32)

    N_large = 100000
    pos64_large = np.zeros((N_large, 2), dtype=np.float64)
    pos32_large = np.zeros((N_large, 2), dtype=np.float32)

    bytes_64_actual = pos64_large.nbytes
    bytes_32_actual = pos32_large.nbytes

    sizeof_double = 8
    sizeof_float = 4
    bytes_64_hand = N_large * 2 * sizeof_double
    bytes_32_hand = N_large * 2 * sizeof_float

    mem_match = (bytes_64_actual == bytes_64_hand) and (bytes_32_actual == bytes_32_hand)
    checks.append(("Memory layout & N=100k scaling", mem_match))
    if verbose:
        status = "OK" if mem_match else "FAIL"
        print(f"  1. Memory layout (float64: {bytes_64_actual/1e6:.2f} MB, float32: {bytes_32_actual/1e6:.2f} MB): {status}")

    # 2. Distance matrix symmetry and diagonal
    rng = np.random.default_rng(101)
    pos_test = rng.uniform(0.0, 10.0, size=(25, 2))
    rc_test = 2.5
    d_mat, dist_mat, neigh_mat = pairwise_distances(pos_test, rc=rc_test)

    sym_diff = float(np.max(np.abs(dist_mat - dist_mat.T)))
    pass_sym = (sym_diff < 1e-14)
    checks.append(("Distance symmetry", pass_sym))

    diag_max = float(np.max(np.abs(np.diag(dist_mat))))
    pass_diag = (diag_max == 0.0)
    checks.append(("Zero diagonal", pass_diag))

    pass_self = not np.any(np.diag(neigh_mat))
    checks.append(("Self-neighbor exclusion", pass_self))

    if verbose:
        print(f"  2. Distance matrix symmetry (residual = {sym_diff:.2e}): {'OK' if pass_sym else 'FAIL'}")
        print(f"  3. Zero diagonal entries: {'OK' if pass_diag else 'FAIL'}")
        print(f"  4. Self-neighbor exclusion: {'OK' if pass_self else 'FAIL'}")

    # 3. 4-particle hand benchmark
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
    checks.append(("4-particle Pythagorean distances", pass_analytic))

    counts_calc = np.sum(neigh_4, axis=1).tolist()
    counts_expected = [2, 1, 1, 0]
    pass_counts = (counts_calc == counts_expected)
    checks.append(("4-particle neighbor counts", pass_counts))

    if verbose:
        print(f"  5. 4-particle Pythagorean benchmark: {'OK' if pass_analytic else 'FAIL'}")
        print(f"  6. 4-particle neighbor counts (rc=2.1): {'OK' if pass_counts else 'FAIL'}")

    # 4. Cutoff scaling ensemble
    pos_50 = rng.uniform(0.0, 10.0, size=(50, 2))
    rc_list = [1.0, 1.5, 2.0, 3.0]
    avg_neighbors = []
    for rc_val in rc_list:
        _, _, n_mat = pairwise_distances(pos_50, rc=rc_val)
        avg_neighbors.append(float(np.mean(np.sum(n_mat, axis=1))))

    pass_monotonic = all(avg_neighbors[i] < avg_neighbors[i+1] for i in range(len(avg_neighbors)-1))
    checks.append(("Cutoff scaling monotonicity", pass_monotonic))

    if verbose:
        print(f"  7. Cutoff scaling monotonicity (rc in [1.0, 1.5, 2.0, 3.0]): {'OK' if pass_monotonic else 'FAIL'}")

    # 5. Newton's Third Law
    F_pair = pair_forces(pos_test, k=15.0, r0=1.5, rc=2.5)
    net_F = np.sum(F_pair, axis=0)
    net_F_norm = float(np.linalg.norm(net_F))
    pass_newton = (net_F_norm < 1e-12)
    checks.append(("Newton's Third Law", pass_newton))

    if verbose:
        print(f"  8. Newton's Third Law (net force = {net_F_norm:.2e} N): {'OK' if pass_newton else 'FAIL'}")

    # 6. Wall reflections and clamping
    pos_outside = np.array([
        [-0.5, 5.0],
        [10.5, 5.0],
        [5.0, -0.5],
        [5.0, 10.5]
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
    checks.append(("Wall reflection boundary condition", pass_walls))

    if verbose:
        print(f"  9. Wall reflection and clamping: {'OK' if pass_walls else 'FAIL'}")

    # 7. Energy positivity and baseline check
    pos_init, vel_init, mass_init = init_state(seed=42, N=12, L=10.0)
    K0 = kinetic_energy(vel_init, mass_init)
    U0 = potential_energy(pos_init, k=15.0, r0=1.5, rc=1.5)
    E0 = K0 + U0

    pass_energy = (K0 > 0) and (U0 >= 0) and abs(E0 - 77.1426) < 0.01
    checks.append(("Energy positivity & E0 baseline", pass_energy))

    if verbose:
        print(f" 10. Initial energy consistency (E0 = {E0:.4f} J): {'OK' if pass_energy else 'FAIL'}")

    passed_count = sum(1 for _, s in checks if s)
    total_count = len(checks)
    print("-" * 50)
    print(f"Result: {passed_count}/{total_count} tests passed.")
    print("-" * 50)

    return passed_count == total_count


def run_task_1(headless=True):
    return verify_task1(verbose=True)


if __name__ == "__main__":
    success = run_task_1()
    sys.exit(0 if success else 1)
