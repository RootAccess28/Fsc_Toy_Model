import numpy as np
import matplotlib.pyplot as plt
import ctypes

# MEMORY AND SCALABILITY - FSC (2D)

PARTICLE_COUNTS = [100, 1000, 5000, 10000]
DIMENSIONS = 2
VERIFY_N = 1000


# --------------------------------------------------
# 1. MEMORY LAYOUT OF POSITION ARRAY
# --------------------------------------------------

np.random.seed(0)

N_initial = 12
pos = np.random.rand(N_initial, 2) * 10
pos32 = pos.astype(np.float32)

print("\nMEMORY LAYOUT OF POSITION ARRAY")
print("--------------------------------")

print("Shape:", pos.shape)
print("Data type:", pos.dtype)
print("Item size:", pos.itemsize, "bytes")
print("Total memory:", pos.nbytes, "bytes")
print("Strides:", pos.strides)

print("\nFLOAT32 POSITION ARRAY")
print("Data type:", pos32.dtype)
print("Item size:", pos32.itemsize, "bytes")
print("Total memory:", pos32.nbytes, "bytes")

print("\nC DATA TYPE SIZES")
c_float_size = ctypes.sizeof(ctypes.c_float)
c_double_size = ctypes.sizeof(ctypes.c_double)

print("sizeof(float):", c_float_size, "bytes")
print("sizeof(double):", c_double_size, "bytes")


# --------------------------------------------------
# 2. POSITION MEMORY FOR N = 100000
# --------------------------------------------------

N_large = 100000

position_memory_64 = (
    N_large * DIMENSIONS * c_double_size
)

position_memory_32 = (
    N_large * DIMENSIONS * c_float_size
)

print("\nPOSITION MEMORY FOR N = 100000")
print("--------------------------------")

print("Float64:", position_memory_64, "bytes")
print("Float32:", position_memory_32, "bytes")

print(
    "Float64:",
    round(position_memory_64 / (1024 ** 2), 3),
    "MiB"
)

print(
    "Float32:",
    round(position_memory_32 / (1024 ** 2), 3),
    "MiB"
)


# --------------------------------------------------
# 3. PAIRWISE DISPLACEMENT MEMORY CALCULATION
# --------------------------------------------------

def calculate_memory(N, dtype):
    itemsize = np.dtype(dtype).itemsize
    return N * N * DIMENSIONS * itemsize


def format_memory(size):
    mib = size / (1024 ** 2)
    gib = size / (1024 ** 3)

    if gib >= 1:
        return f"{gib:.3f} GiB"

    return f"{mib:.3f} MiB"


print("\nPAIRWISE DISPLACEMENT MEMORY")
print("Array shape: (N, N, 2)")
print("Formula: N x N x 2 x itemsize\n")

print(
    f"{'N':>8} | {'Float32':>15} | "
    f"{'Float64':>15}"
)
print("-" * 45)

for N in PARTICLE_COUNTS:

    memory32 = calculate_memory(N, np.float32)
    memory64 = calculate_memory(N, np.float64)

    print(
        f"{N:8d} | "
        f"{format_memory(memory32):>15} | "
        f"{format_memory(memory64):>15}"
    )


# --------------------------------------------------
# 4. VERIFY THEORETICAL MEMORY WITH NUMPY NBYTES
# --------------------------------------------------

print("\nNUMPY MEMORY VERIFICATION")
print("Verification N =", VERIFY_N)

for dtype in [np.float32, np.float64]:

    displacement = np.zeros(
        (VERIFY_N, VERIFY_N, DIMENSIONS),
        dtype=dtype
    )

    theoretical = calculate_memory(VERIFY_N, dtype)

    print("\nData type:", displacement.dtype)
    print("Shape:", displacement.shape)
    print("Item size:", displacement.itemsize, "bytes")
    print("Theoretical memory:", theoretical, "bytes")
    print("NumPy nbytes:", displacement.nbytes, "bytes")
    print("Match:", theoretical == displacement.nbytes)

    del displacement


# --------------------------------------------------
# 5. GRAPH 1: MEMORY VS PARTICLE COUNT
# --------------------------------------------------

N_values = np.array(PARTICLE_COUNTS)

memory32 = np.array([
    calculate_memory(N, np.float32) / (1024 ** 3)
    for N in PARTICLE_COUNTS
])

memory64 = np.array([
    calculate_memory(N, np.float64) / (1024 ** 3)
    for N in PARTICLE_COUNTS
])

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(
    N_values,
    memory32,
    marker="o",
    markersize=8,
    linewidth=2.5,
    label="Float32"
)

ax.plot(
    N_values,
    memory64,
    marker="s",
    markersize=8,
    linewidth=2.5,
    label="Float64"
)

ax.set_title(
    "Memory Requirement vs Number of Particles (2D)",
    fontsize=14,
    fontweight="bold",
    pad=15
)

ax.set_xlabel("Number of Particles (N)", fontsize=12)
ax.set_ylabel("Memory Requirement (GiB)", fontsize=12)

# Linear particle-count axis and logarithmic memory axis
ax.set_yscale("log")
ax.set_xticks(N_values)
ax.set_xticklabels([f"{n:,}" for n in N_values])

ax.grid(
    True,
    which="both",
    linestyle="--",
    alpha=0.4
)

ax.legend(fontsize=11)
ax.margins(x=0.08)

fig.tight_layout()

fig.savefig(
    "memory_and_scalability_fsc.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# --------------------------------------------------
# 6. GRAPH 2: MEMORY INCREASE BETWEEN PARTICLE COUNTS
# --------------------------------------------------

growth_factors = (
    N_values[1:] / N_values[:-1]
) ** 2

interval_labels = [
    f"{N_values[i]:,} to {N_values[i + 1]:,}"
    for i in range(len(N_values) - 1)
]

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(
    interval_labels,
    growth_factors,
    width=0.55
)

ax.set_title(
    "Increase in Memory Between Particle Counts (2D)",
    fontsize=14,
    fontweight="bold",
    pad=15
)

ax.set_xlabel("Increase in Particle Count", fontsize=12)
ax.set_ylabel("Memory Multiplication Factor", fontsize=12)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

ax.set_axisbelow(True)

for bar, value in zip(bars, growth_factors):

    ax.annotate(
        f"{value:.0f}x",
        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
        xytext=(0, 7),
        textcoords="offset points",
        ha="center",
        fontsize=11,
        fontweight="bold"
    )

ax.set_ylim(0, max(growth_factors) * 1.2)

fig.tight_layout()

fig.savefig(
    "memory_growth_fsc.png",
    dpi=300,
    bbox_inches="tight"
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

