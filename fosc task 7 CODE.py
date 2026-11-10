<<<<<<< HEAD
"""
Task 7: Memory and Scalability Analysis (Backward-Compatibility Shim)
=====================================================================
Redirects to Task7_Memory_Scalability/task7_memory.py.
Preserves existing import and execution paths.
"""

from Task7_Memory_Scalability.task7_memory import (
    calculate_memory,
    run_task_7,
    DEFAULT_N_VALUES,
    DATA_TYPES,
)

if __name__ == "__main__":
    run_task_7()
=======
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

N_VALUES = [100, 1000, 5000, 10000]
DIMENSIONS = 3
VERIFY_N = 1000

DATA_TYPES = {
    "float32": np.float32,
    "float64": np.float64
}


# Memory of one displacement array
def displacement_memory(n, dtype):
    return n * n * DIMENSIONS * np.dtype(dtype).itemsize


# Approximate memory for vectorized working arrays
def working_memory(n, dtype):
    itemsize = np.dtype(dtype).itemsize
    bytes_per_pair = 11 * itemsize + 1
    return n * n * bytes_per_pair


def to_gib(byte_count):
    return byte_count / (1024 ** 3)


# 1. Memory table
def print_memory_table():

    print("\n3D MEMORY AND SCALABILITY")
    print("Shape = (N, N, 3)")
    print("Formula = N x N x 3 x itemsize\n")

    print(f"{'N':>8} | {'float32':>12} | {'float64':>12}")
    print("-" * 40)

    for n in N_VALUES:

        m32 = to_gib(displacement_memory(n, np.float32))
        m64 = to_gib(displacement_memory(n, np.float64))

        print(f"{n:8d} | {m32:9.4f} GiB | {m64:9.4f} GiB")


# 2. NumPy verification
def verify_numpy():

    print("\nNUMPY nbytes VERIFICATION")

    for dtype in DATA_TYPES.values():

        array = np.empty(
            (VERIFY_N, VERIFY_N, DIMENSIONS),
            dtype=dtype
        )

        expected = displacement_memory(VERIFY_N, dtype)

        print("\nData type:", np.dtype(dtype))
        print("Shape:", array.shape)
        print("Theoretical:", expected, "bytes")
        print("NumPy nbytes:", array.nbytes)
        print("Match:", expected == array.nbytes)

        del array


# 3. Scalability estimates
def print_scalability():

    print("\nESTIMATED VECTORISED WORKING MEMORY")
    print("These are approximate estimates, not measured RAM usage.")

    print(f"{'N':>8} | {'float32':>12} | {'float64':>12}")
    print("-" * 40)

    for n in N_VALUES:

        m32 = to_gib(working_memory(n, np.float32))
        m64 = to_gib(working_memory(n, np.float64))

        print(f"{n:8d} | {m32:9.4f} GiB | {m64:9.4f} GiB")

    print("\nMemory grows approximately as O(N^2).")
    print("Doubling N makes memory four times larger.")


# GRAPH 1: Displacement memory
def graph_displacement():

    plt.figure(figsize=(9, 6))

    for name, dtype in DATA_TYPES.items():

        values = [
            to_gib(displacement_memory(n, dtype))
            for n in N_VALUES
        ]

        plt.plot(
            N_VALUES,
            values,
            marker="o",
            linewidth=2,
            markersize=7,
            label=name
        )

    plt.xlabel("Number of particles (N)")
    plt.ylabel("Displacement memory (GiB)")
    plt.title("3D Pairwise Displacement Memory vs Particle Count")

    plt.xticks(N_VALUES)
    plt.xlim(0, 11000)
    plt.yscale("log")

    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig("task7_graph1_displacement_memory.png", dpi=300)
    plt.show()


# GRAPH 2: Interactive working memory comparison
def graph_interactive():

    fig, ax = plt.subplots(figsize=(9, 6))

    fig.subplots_adjust(bottom=0.25, top=0.82)

    initial_n = 10000

    values = [
        to_gib(working_memory(initial_n, np.float32)),
        to_gib(working_memory(initial_n, np.float64))
    ]

    bars = ax.bar(
        ["float32", "float64"],
        values,
        width=0.5
    )

    ax.set_yscale("log")
    ax.set_ylabel("Estimated working memory (GiB)")
    ax.set_title(
        "Estimated 3D Vectorized Working Memory\n"
        f"Particle Count N = {initial_n}"
    )

    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    # Display values above bars
    labels = []

    for bar, value in zip(bars, values):

        label = ax.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.15,
            f"{value:.4f} GiB",
            ha="center",
            fontsize=10
        )

        labels.append(label)

    ax.set_ylim(min(values) / 2, max(values) * 3)

    # Slider for particle count
    slider_ax = fig.add_axes([0.20, 0.10, 0.65, 0.04])

    particle_slider = Slider(
        slider_ax,
        "Particle count",
        0,
        len(N_VALUES) - 1,
        valinit=3,
        valstep=1
    )

    def update(value):

        index = int(value)
        n = N_VALUES[index]

        new_values = [
            to_gib(working_memory(n, np.float32)),
            to_gib(working_memory(n, np.float64))
        ]

        for bar, label, new_value in zip(
            bars, labels, new_values
        ):

            bar.set_height(new_value)

            label.set_y(new_value * 1.15)
            label.set_text(f"{new_value:.4f} GiB")

        ax.set_ylim(
            min(new_values) / 2,
            max(new_values) * 3
        )

        ax.set_title(
            "Estimated 3D Vectorized Working Memory\n"
            f"Particle Count N = {n}"
        )

        fig.canvas.draw_idle()

    particle_slider.on_changed(update)

    plt.savefig(
        "task7_graph2_interactive_memory.png",
        dpi=300
    )

    plt.show()


# MAIN PROGRAM
def main():

    print("NumPy version:", np.__version__)

    print_memory_table()
    verify_numpy()
    print_scalability()

    print("\nOpening Graph 1...")
    graph_displacement()

    print("\nOpening Interactive Graph 2...")
    graph_interactive()

    print("\nBoth graphs saved successfully!")
    print("TASK 7 COMPLETED.")


if __name__ == "__main__":
    main()

>>>>>>> FORK/main
