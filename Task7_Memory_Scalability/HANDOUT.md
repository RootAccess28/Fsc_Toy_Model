# Task 7: Memory and Scalability Analysis (2D)

## 1. What This Task Does
Task 7 examines how memory requirements scale as the particle count ($N$) grows in our 2D molecular simulation. It explores the critical difference between linear memory $\mathcal{O}(N)$ for particle positions and quadratic memory $\mathcal{O}(N^2)$ for pairwise displacement arrays.

We analyze:
1. Exact memory layout of 2D arrays (strides, itemsize, data types).
2. Theoretical memory scaling vs actual NumPy `array.nbytes` allocations.
3. The memory reduction achieved by switching from `float64` to `float32`.
4. The exact particle count where dense NumPy broadcasting triggers an Out-Of-Memory (OOM) crash on consumer workstations.

---

## 2. Core Mathematical & Computational Concepts

### 2.1 The Two Fundamental Memory Regimes
- **1. Particle State Arrays (Linear Scaling, $\mathcal{O}(N)$)**:
  - Position array shape: $(N, 2)$ where each particle has coordinates $(x, y)$.
  - Velocity array shape: $(N, 2)$
  - Memory formula:
    $$M_{\text{pos}} = N \times 2 \times \text{itemsize}$$
  - For $N = 100,000$ particles:
    - `float64` (8 bytes): $100,000 \times 2 \times 8 = 1.60\text{ MB}$ ($1.526\text{ MiB}$)
    - `float32` (4 bytes): $100,000 \times 2 \times 4 = 0.80\text{ MB}$ ($0.763\text{ MiB}$)
  - *Conclusion*: Storing particle coordinates is very cheap; millions of particles fit easily into RAM.

- **2. Pairwise Displacement Tensor (Quadratic Scaling, $\mathcal{O}(N^2)$)**:
  - To compute distances between all pairs simultaneously in vectorized NumPy, the code creates an $(N, N, 2)$ array:
    $$\mathbf{d}_{ij} = \mathbf{r}_i - \mathbf{r}_j$$
  - Memory formula:
    $$M_{\text{disp}} = N^2 \times 2 \times \text{itemsize}$$
  - Because of the $N^2$ term, memory demand explodes rapidly.

### 2.2 Theoretical vs NumPy `nbytes` Verification ($N = 1,000$)
To verify theoretical math against actual runtime memory, we allocate test arrays in NumPy and compare `calculate_memory(N, dtype)` against `arr.nbytes`:

| Data Type | Item Size | Theoretical Bytes | NumPy `arr.nbytes` | Exact Match? |
| :--- | :--- | :--- | :--- | :--- |
| **`float32`** | 4 bytes | $1000^2 \times 2 \times 4 = 8,000,000\text{ B}$ | $8,000,000\text{ B}$ | **True** |
| **`float64`** | 8 bytes | $1000^2 \times 2 \times 8 = 16,000,000\text{ B}$ | $16,000,000\text{ B}$ | **True** |

### 2.3 Quadratic Memory Growth Factors
When scaling particle counts:
$$\text{Growth Factor} = \left(\frac{N_{\text{new}}}{N_{\text{old}}}\right)^2$$
- $N = 100 \to 1,000$ ($10\times$ particles): Memory increases by $10^2 = \mathbf{100\times}$ ($0.15\text{ MiB} \to 15.26\text{ MiB}$).
- $N = 1,000 \to 5,000$ ($5\times$ particles): Memory increases by $5^2 = \mathbf{25\times}$ ($15.26\text{ MiB} \to 381.47\text{ MiB}$).
- $N = 5,000 \to 10,000$ ($2\times$ particles): Memory increases by $2^2 = \mathbf{4\times}$ ($381.47\text{ MiB} \to 1.49\text{ GiB}$).

### 2.4 The Hardware Out-Of-Memory (OOM) Limit ($N = 50,000$)
At $N = 50,000$ particles:
- `float32`: $50,000^2 \times 2 \times 4\text{ B} = 20,000,000,000\text{ B} \approx \mathbf{18.63\text{ GiB}}$
- `float64`: $50,000^2 \times 2 \times 8\text{ B} = 40,000,000,000\text{ B} \approx \mathbf{37.25\text{ GiB}}$

On standard consumer laptops (8 GB to 16 GB RAM), allocating this single array triggers an immediate Operating System Out-Of-Memory crash or stalls the machine in disk swapping.

*Why This Matters*: This experiment demonstrates why large-scale production Molecular Dynamics simulators (LAMMPS, GROMACS, NAMD) never store full $N \times N$ pairwise arrays. Instead, they divide the simulation domain into spatial bins (cell lists / neighbor lists) that reduce memory and computation to linear $\mathcal{O}(N)$ scaling!

---

## 3. How to Run the Memory Experiment

Run the script from the terminal:
```bash
# Run Task 7 directly:
python3 Task7_Memory_Scalability/task7_memory.py

# Or via the friend's script name:
python3 memory_and_scalability_fsc.py
# or: python3 "fosc task 7 CODE.py"

# Or via the master runner:
python3 simulator.py --task7
```

### Generated Output Files in `task7_results/`
1. **`memory_measurements.csv`**: Measured byte and MiB sizes of position arrays across particle scales.
2. **`pairwise_memory_estimates.csv`**: Theoretical memory requirements for $(N, N, 2)$ displacement arrays.
3. **`task7_environment.txt`**: Complete system environment specs (Python, NumPy, Matplotlib, OS, processor).
4. **`position_memory_comparison.png`**: Bar chart comparing `float32` vs `float64` position memory footprints.
5. **`pairwise_memory_scalability.png`**: Log-log curve of pairwise memory vs $N$, featuring dashed horizontal threshold lines for 8 GB, 16 GB, and 32 GB RAM hardware limits.
6. **`memory_growth_fsc.png`**: Bar chart showing the memory multiplication factors ($100\times$, $25\times$, $4\times$).

---

## 4. Key Questions You Might Be Asked

**Q1: Why is the position array shape $(N, 2)$ while the displacement array is $(N, N, 2)$?**
> *Answer*: The position array only needs to store the current $(x, y)$ coordinate for each of the $N$ particles. The displacement array stores the vector difference $\mathbf{r}_i - \mathbf{r}_j$ between *every pair* of particles $i$ and $j$, requiring an $N \times N$ grid of 2D vectors.

**Q2: Why do we estimate large pairwise memory mathematically instead of allocating it?**
> *Answer*: At $N = 50,000$ or $N = 100,000$, allocating a $(100000, 100000, 2)$ float64 array would require nearly $150\text{ GiB}$ of RAM. Calculating it mathematically using $N^2 \times 2 \times \text{itemsize}$ allows us to accurately predict hardware limits without crashing our computer.

**Q3: How does choosing `float32` help with memory scalability?**
> *Answer*: `float32` cuts array memory exactly in half ($50\%$ savings) because each number takes 4 bytes instead of 8 bytes. This allows a simulation to handle approximately $\sqrt{2} \approx 1.41\times$ more particles within the same physical RAM budget.
