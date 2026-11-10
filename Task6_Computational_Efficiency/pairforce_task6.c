/*
 * Task 6 - compiled C pair-force kernel (2D)
 * Adapted from the group's uploaded pairforce.c.
 *
 * ABI retained from the uploaded source:
 *   compute_pair_forces(N, pos, F, rc, r0, k)
 *
 * The 1e-12 denominator offset matches task3_4.py's NumPy unit-vector
 * calculation: d / (dist + 1e-12). Positions and forces are contiguous
 * row-major double arrays with shape (N, 2).
 */
#include <math.h>

void compute_pair_forces(int N,
                         const double *pos,
                         double *F,
                         double rc,
                         double r0,
                         double k) {
    const double eps = 1.0e-12;

    for (int i = 0; i < 2 * N; ++i) {
        F[i] = 0.0;
    }

    /* Visit each unordered pair exactly once: j > i. */
    for (int i = 0; i < N; ++i) {
        const double xi = pos[2 * i];
        const double yi = pos[2 * i + 1];

        for (int j = i + 1; j < N; ++j) {
            const double dx = xi - pos[2 * j];
            const double dy = yi - pos[2 * j + 1];
            const double dist = sqrt(dx * dx + dy * dy);

            if (dist > eps && dist <= rc) {
                const double factor = -k * (dist - r0) / dist;
                const double fx = factor * dx;
                const double fy = factor * dy;

                F[2 * i]     += fx;
                F[2 * i + 1] += fy;
                F[2 * j]     -= fx;
                F[2 * j + 1] -= fy;
            }
        }
    }
}
