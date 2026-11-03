/**
 * Fundamentals of Scientific Computing: Atomic Toy Model Project
 * Task 4 & Task 6: Compiled C Pair Force Calculation
 * Compile with: gcc -O2 pairforce.c -o pairforce -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* 
 * Computes pairwise spring forces between particles using an optimized O(N^2/2) 
 * double loop exploiting Newton's Third Law (F_ji = -F_ij).
 */
void compute_pair_forces(int N, const double *pos, double *F, double rc, double r0, double k) {
    // 1. Initialize all force components to zero
    for (int i = 0; i < N * 2; i++) {
        F[i] = 0.0;
    }

    // 2. Double loop over unique pairs (j > i)
    for (int i = 0; i < N; i++) {
        double xi = pos[2 * i];
        double yi = pos[2 * i + 1];

        for (int j = i + 1; j < N; j++) {
            double xj = pos[2 * j];
            double yj = pos[2 * j + 1];

            double dx = xi - xj;
            double dy = yi - yj;
            double dist = sqrt(dx * dx + dy * dy);

            // Apply spring force only within cutoff rc and exclude self-pairing
            if (dist > 1e-12 && dist <= rc) {
                double fmag = -k * (dist - r0);
                double ux = dx / dist;
                double uy = dy / dist;

                // Force vector components
                double fx = fmag * ux;
                double fy = fmag * uy;

                // Equal and opposite forces (Newton's 3rd Law)
                F[2 * i]     += fx;
                F[2 * i + 1] += fy;

                F[2 * j]     -= fx;
                F[2 * j + 1] -= fy;
            }
        }
    }
}

int main(void) {
    // Basic verification test on N=4 particles
    int N = 4;
    double rc = 2.0;
    double r0 = 0.8;
    double k = 15.0;

    double pos[8] = {
        1.0, 1.0,
        2.0, 1.5,
        3.0, 3.0,
        1.5, 2.5
    };
    double F[8];

    compute_pair_forces(N, pos, F, rc, r0, k);

    printf("=== C PAIRFORCE VERIFICATION TEST (N=4) ===\n");
    for (int i = 0; i < N; i++) {
        printf("Particle %d: Fx = %10.6f, Fy = %10.6f\n", i, F[2 * i], F[2 * i + 1]);
    }

    // Verify total force balance (norm of sum of all forces ~ 0)
    double sum_fx = 0.0, sum_fy = 0.0;
    for (int i = 0; i < N; i++) {
        sum_fx += F[2 * i];
        sum_fy += F[2 * i + 1];
    }
    printf("Total Internal Force Vector: (%.2e, %.2e)\n", sum_fx, sum_fy);

    return 0;
}