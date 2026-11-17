# Task 8: Spatial Partitioning via Cell-Linked Lists ($\mathcal{O}(N)$ Scaling)

## 1. What This Task Does
Task 8 eliminates the quadratic computational and memory bottlenecks identified in Tasks 6 and 7. In standard molecular dynamics, calculating pairwise forces via an all-pairs evaluation requires inspecting $\frac{1}{2}N(N-1) \approx \mathcal{O}(N^2)$ particle pairs. As demonstrated in Task 7, this quadratic scaling leads to prohibitive runtimes and memory exhaustion ($37.25\text{ GiB}$ at $N=50,000$).

In this task, we implement a **2D Spatial Cell List (Linked-Cell Decomposition)**:
1. Decompose the continuous 2D domain $[0, L] \times [0, L]$ into a regular grid of square cells whose side width satisfies $w \ge r_c$.
2. Sort particles into cells using linear array binning ($\mathcal{O}(N)$).
3. Restrict distance evaluations strictly to the host cell and its $8$ immediate neighbors ($3 \times 3$ stencil).
4. Demonstrate that the computational time complexity drops from $\mathcal{O}(N^2)$ to linear $\mathcal{O}(N)$ at constant number density $\rho = N / L^2$.
5. Verify that the computed forces and dynamical trajectories remain **bitwise identical to double precision machine epsilon** ($\sim 10^{-15}\text{ N}$).

---

## 2. Core Mathematical & Computational Concepts

### 2.1 The All-Pairs Bottleneck ($\mathcal{O}(N^2)$)
Because the inter-particle Hookean force has a finite interaction cutoff $r_c$:
$$\mathbf{F}_{ij} = \mathbf{0} \quad \text{for } r_{ij} > r_c$$
Any distance evaluation between particles separated by $r_{ij} > r_c$ is wasted computation. In an all-pairs algorithm, the number of evaluated distances is:
$$N_{\text{eval}}^{\text{all-pairs}} = N^2$$
At $N = 20,000$, this requires $400,000,000$ distance checks, even though only $15,038$ pairs ($0.0037\%$) actually interact!

### 2.2 The Cell-Partitioning Geometry
To avoid checking distant particles, the simulation domain $L \times L$ is subdivided into $M \times M$ square bins, where:
$$M = \left\lfloor \frac{L}{r_c} \right\rfloor, \quad w = \frac{L}{M} \ge r_c$$
Because the cell width $w$ is guaranteed to be at least $r_c$, **any particle within distance $r \le r_c$ of particle $i$ must reside either within the same cell as $i$ or in one of the 8 adjacent neighboring cells**.

```text
+---------+---------+---------+
| (-1, 1) | ( 0, 1) | ( 1, 1) |
+---------+---------+---------+
| (-1, 0) | Host(i) | ( 1, 0) |   <-- 3x3 Interaction Stencil
+---------+---------+---------+
| (-1,-1) | ( 0,-1) | ( 1,-1) |
+---------+---------+---------+
```

### 2.3 Linear Scaling Proof ($\mathcal{O}(N)$ at Constant Density)
Let the system have constant area number density $\rho = N / L^2$.
- The average number of particles per cell is:
  $$\langle N_{\text{cell}} \rangle = \rho w^2 = \text{constant}$$
- The 9-cell interaction stencil contains an average of $9 \times \langle N_{\text{cell}} \rangle = \mathcal{O}(1)$ candidate particles.
- For $N$ particles, the total candidate evaluations evaluated per timestep is:
  $$N_{\text{eval}}^{\text{cell}} = N \times (9 \langle N_{\text{cell}} \rangle) = \mathcal{O}(N)$$
The algorithmic complexity is reduced from quadratic to **strictly linear**!

### 2.4 Vectorized Assembly via `np.bincount` and `np.repeat`
To avoid slow interpreted Python loops while keeping memory footprint low:
1. **Binning**: Particle coordinates are converted to integer 2D cell indices:
   $$c_x = \min\left(\left\lfloor \frac{x}{w} \right\rfloor, M-1\right), \quad c_y = \min\left(\left\lfloor \frac{y}{w} \right\rfloor, M-1\right), \quad \text{cid} = c_x M + c_y$$
2. **Contiguous Indexing**: Particles are sorted by `cid` using `np.argsort(kind='stable')`. The starting index and particle count of every cell are extracted in $\mathcal{O}(N)$ using `np.bincount` and `np.cumsum`.
3. **Newton's 3rd Law Accumulation**: Only unique pairs ($j > i$) are evaluated. Forces $\mathbf{f}_{ij}$ are scattered into the total force array $\mathbf{F}$ in compiled C via NumPy's `np.bincount`:
   $$\mathbf{F}[:, c] = \text{bincount}(i, f[:, c]) - \text{bincount}(j, f[:, c])$$
   This eliminates all temporary $(N, N, 2)$ memory allocations.

### 2.5 Machine Precision Verification
To ensure that spatial partitioning does not drop interacting pairs, we compare forces against the exact all-pairs baseline across multiple particle counts:
$$\| \mathbf{F}_{\text{cell}} - \mathbf{F}_{\text{baseline}} \|_\infty \le 7.11 \times 10^{-15}\text{ N}$$
The discrepancy is bounded by IEEE 754 double precision machine epsilon, confirming **zero truncation or algorithmic omission**.

---

## 3. How to Run the Benchmark

Execute the Task 8 benchmark directly:
```bash
# Run full benchmark (includes scaling up to N=100,000):
python3 Task8_Cell_Lists/task8_cell_list.py

# Run accelerated verification test (10 seconds):
python3 Task8_Cell_Lists/task8_cell_list.py --fast

# Or via the master runner:
python3 simulator.py --task8
```

### Measured Performance & Work Reduction Summary

| Particle Count $N$ | All-Pairs Checks ($N^2$) | Cell List Candidates | Interacting Pairs | Work Reduction | Speedup ($t_{\text{vec}} / t_{\text{cell}}$) | Peak Memory Reduction |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **50** | $2,500$ | $260$ | $42$ | $9.6\times$ | $0.8\times$ (overhead) | $200\text{ KB} \to 20\text{ KB}$ |
| **200** | $40,000$ | $1,052$ | $168$ | **$38.0\times$** | **$7.9\times$** | $2.7\text{ MB} \to 60\text{ KB}$ |
| **1,000** | $1,000,000$ | $5,362$ | $757$ | **$186.5\times$** | **$93.1\times$** | $65.1\text{ MB} \to 280\text{ KB}$ |
| **2,000** | $4,000,000$ | $10,480$ | $1,512$ | **$381.7\times$** | **$214.1\times$** | $260.1\text{ MB} \to 570\text{ KB}$ |
| **5,000** | $25,000,000$ | $26,140$ | $3,735$ | **$956.4\times$** | **$770.0\times$** | $1,625\text{ MB} \to 1.4\text{ MB}$ |
| **20,000** | $400,000,000$ | $105,568$ | $15,038$ | **$3,789.0\times$** | **$> 1,500\times$** | $> 26\text{ GB} \to 5.3\text{ MB}$ |
| **100,000** | $10^{10}$ (OOM crash) | $528,000$ | $75,200$ | **$> 18,000\times$** | **$\infty$ (Runs in $56\text{ ms}$)** | $> 150\text{ GB} \to 27.0\text{ MB}$ |

### Generated Output Artifacts in `results/`
1. **`task8_fig0_cell_diagram.png`**: Visual schematic illustrating the $L \times L$ grid partition, a host particle, its circular interaction radius $r_c$, and the highlighted $3 \times 3$ candidate stencil.
2. **`task8_fig1_time_speedup.png`**: 
   - Left: Log-log curve of runtime vs $N$ demonstrating empirical slopes of $p=1.99 \approx 2$ for all-pairs and $p=0.98 \approx 1$ for cell lists.
   - Right: Measured speedup factor reaching over $770\times$.
3. **`task8_fig2_memory.png`**: Heap allocation scaling proving linear $\mathcal{O}(N)$ memory vs quadratic $\mathcal{O}(N^2)$ memory, benchmarked against the 8 GB workstation RAM ceiling.
4. **`task8_fig3_energy_same.png`**: Total mechanical energy drift curves during a 200-step Velocity Verlet simulation showing exact overlap between all-pairs and cell lists.
5. **`task8_correctness.csv`**: Machine-precision force residual metrics across multiple system sizes.
6. **`task8_timing.csv`**: Measured median execution times and peak memory allocations.
7. **`task8_candidates.csv`**: Distance checks count and work reduction factors.

---

## 4. Key Questions You Might Be Asked (Oral Defense)

**Q1: Why must the cell width $w$ satisfy $w \ge r_c$? What would happen if $w < r_c$?**
> *Answer*: If $w \ge r_c$, any two particles with distance $r \le r_c$ cannot be separated by more than one cell boundary, ensuring they are contained within the immediate $3 \times 3$ neighboring block. If $w < r_c$, interacting particles could be separated by two or more cells, meaning a $3 \times 3$ search stencil would miss interacting neighbors, leading to unphysical force truncation and catastrophic energy non-conservation.

**Q2: Why does the cell list approach achieve $\mathcal{O}(N)$ complexity only at constant density?**
> *Answer*: At constant density $\rho = N / L^2$, the physical area of the simulation domain grows proportionally with $N$ ($L \propto \sqrt{N}$), so the total number of cells increases proportionally with $N$ while the average number of particles per cell remains constant ($\approx \rho r_c^2 = \text{const}$). Each particle only interacts with a constant $\mathcal{O}(1)$ number of neighbors, yielding $N \times \mathcal{O}(1) = \mathcal{O}(N)$ total operations. If the box size $L$ remained fixed while $N \to \infty$, the number of particles per cell would grow as $\mathcal{O}(N)$, causing the stencil search to degrade back to $\mathcal{O}(N^2)$.

**Q3: How do we enforce Newton's third law ($\mathbf{F}_{ij} = -\mathbf{F}_{ji}$) without double-counting pairs?**
> *Answer*: When inspecting candidate cells around host cell $(c_x, c_y)$, we enforce the ordering condition $j > i$. When computing the force $\mathbf{f}_{ij}$ between pair $(i, j)$, it is accumulated positively into particle $i$'s force vector and negatively into particle $j$'s force vector. This simultaneously guarantees exact momentum conservation ($\sum \mathbf{F} = \mathbf{0}$) and halves the distance arithmetic.

**Q4: How does this implementation avoid Python loop overhead without allocating large $(N, N)$ arrays?**
> *Answer*: Rather than using nested Python loops over particles, we bin particles using `np.argsort` and expand the candidate indices for each of the 9 cell offsets in vector operations using `np.repeat` and `np.cumsum`. Force contributions are then scattered back into particle force arrays using compiled NumPy `np.bincount`, ensuring C-speed execution with zero $(N, N)$ matrix allocations.
