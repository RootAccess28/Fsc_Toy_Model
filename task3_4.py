"""
Backward-Compatibility Bridge for task3_4
=========================================
Exports:
  - All core physics functions from core.physics
  - Task 3 runner (execute_task_3) from Task3_Timestep_Accuracy
  - Task 4 runner (execute_task_4) from Task4_Viscous_Dissipation
Allows existing scripts or commands referencing task3_4 to continue working seamlessly.
"""

from core.physics import (
    init_state,
    pairwise_distances,
    pair_forces,
    total_forces,
    kinetic_energy,
    potential_energy,
    walls_reflect,
    particle_collisions,
    step_euler,
    step_verlet,
    run_sim,
)
from Task3_Timestep_Accuracy.task3_accuracy import execute_task_3
from Task4_Viscous_Dissipation.task4_dissipation import execute_task_4, find_relaxation_times

# Top-level default constants for full compatibility with legacy code
N = 12
L = 10.0
k = 15.0
rc = 1.5
r0 = 1.5
T_task3 = 10.0
T_task4 = 30.0

if __name__ == "__main__":
    execute_task_3()
    execute_task_4()
    print("ALL PLOTS AND CSV TABLES SUCCESSFULLY SAVED!")
