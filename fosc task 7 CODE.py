import numpy as np
import matplotlib.pyplot as plt

# Particle counts
N_values = [100, 1000, 5000, 10000]

# Data types
data_types = {
    "float32": np.float32,
    "float64": np.float64
}

# 3D displacement memory calculation
def calculate_memory(N, dtype):
    return N * N * 3 * np.dtype(dtype).itemsize


results = {}

print("TASK 7: 3D MEMORY AND SCALABILITY ANALYSIS")
print("=" * 55)

# Theoretical memory calculation
for dtype_name, dtype in data_types.items():

    print("\nData type:", dtype_name)

    for N in N_values:

        memory = calculate_memory(N, dtype)
        memory_MB = memory / (1024 ** 2)

        results[(dtype_name, N)] = memory_MB

        print(
            "N =", N,
            "| Memory =", round(memory_MB, 2), "MB"
        )


# NumPy nbytes verification
print("\nNUMPY nbytes VERIFICATION")
print("=" * 55)

N_test = 1000

for dtype_name, dtype in data_types.items():

    array = np.empty((N_test, N_test, 3), dtype=dtype)

    theoretical = calculate_memory(N_test, dtype)
    actual = array.nbytes

    print("\n", dtype_name)
    print("Theoretical:", theoretical, "bytes")
    print("NumPy nbytes:", actual, "bytes")
    print("Match:", theoretical == actual)

    del array


# Graph 1: Memory vs particle count
plt.figure(figsize=(9, 6))

for dtype_name in data_types:

    memory_values = [
        results[(dtype_name, N)]
        for N in N_values
    ]

    plt.plot(
        N_values,
        memory_values,
        marker="o",
        label=dtype_name
    )

plt.xlabel("Number of particles (N)")
plt.ylabel("Memory required (MB)")
plt.title("3D Memory Requirement vs Number of Particles")
plt.yscale("log")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("task7_memory_scalability_3d.png")
plt.show()


# Graph 2: float32 vs float64
N_compare = 10000

memory_values = [
    calculate_memory(N_compare, dtype) / (1024 ** 2)
    for dtype in data_types.values()
]

plt.figure(figsize=(7, 5))

plt.bar(
    ["float32", "float64"],
    memory_values
)

plt.ylabel("Memory required (MB)")
plt.title("3D Memory Comparison at N = 10000")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("task7_precision_comparison_3d.png")
plt.show()


print("\nGraphs saved successfully!")
print("completed.")