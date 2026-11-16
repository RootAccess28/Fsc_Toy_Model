"""
Task 7: Memory and Scalability Analysis (FSC Toy Model)
=======================================================
Master Task 7 module routing directly to Task7_Memory_Scalability/task7_memory.py.
Outputs (including memory_and_scalability_fsc.png, memory_growth_fsc.png, etc.)
are saved cleanly inside Task7_Memory_Scalability/task7_results/.
"""

import sys
from pathlib import Path

# Ensure root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from Task7_Memory_Scalability.task7_memory import (
    calculate_memory,
    format_memory,
    print_memory_layout,
    print_large_position_memory,
    print_pairwise_memory_table,
    verify_numpy_memory,
    print_oom_analysis,
    generate_plots,
    run_task_7,
    PARTICLE_COUNTS,
    DATA_TYPES,
)

if __name__ == "__main__":
    run_task_7()
