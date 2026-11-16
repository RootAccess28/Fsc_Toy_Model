# Scientific Computing Laboratory Report
## 2D Molecular Dynamics Toy Model: Integration, Stability, and Scalability

**Course:** Fundamentals of Scientific Computing  
**Project:** Molecular / Atomic Interaction Simulation Engine (2D)  
**Authors:** Student Project Team  

---

### Abstract
This report documents the numerical modeling, verification, and computational profiling of a two-dimensional interacting particle system confined to a square domain $[0, L] \times [0, L]$. We investigate:
1. Exact memory layout and stride arithmetic of multidimensional NumPy arrays compared to C pointer offsets.
2. Symplectic time-integration (Velocity Verlet) versus non-symplectic integration (Forward Euler) under varying timesteps $\Delta t$.
3. Energy dissipation rates under Stokes drag ($F_{\text{diss}} = -\gamma v$) and convergence to mechanical equilibrium.
4. Single-precision (`float32`) versus double-precision (`float64`) numerical stability, memory bandwidth, and Lyapunov trajectory divergence.
5. Algorithmic efficiency across interpreted Python, vectorized NumPy broadcasting, and native compiled C (`-O2`).
6. Quadratic memory scaling $\mathcal{O}(N^2)$ of dense pairwise displacement tensors and hardware memory limits.

---

## 1. Governing Physical Model

### 1.1 Equations of Motion
For a system of $N$ particles with masses $m_i$, positions $\mathbf{r}_i \in \mathbb{R}^2$, and velocities $\mathbf{v}_i \in \mathbb{R}^2$, Newton's second law reads:
$$m_i \frac{d^2 \mathbf{r}_i}{dt^2} = \mathbf{F}_i^{\text{internal}} + \mathbf{F}_i^{\text{diss}}$$

### 1.2 Inter-Particle Harmonic Force
Particles interact via a pairwise repulsive linear spring force active within a cutoff radius $r_c$:
$$\mathbf{F}_{ij} = -k (r_{ij} - r_0) \frac{\mathbf{r}_i - \mathbf{r}_j}{r_{ij}} \quad \text{for } 0 < r_{ij} \le r_c$$
where $r_{ij} = \|\mathbf{r}_i - \mathbf{r}_j\|$, $r_0$ is the equilibrium rest length, and $k$ is the spring stiffness. By antisymmetry ($\mathbf{F}_{ij} = -\mathbf{F}_{ji}$), internal momentum is strictly conserved:
$$\sum_{i=1}^N \mathbf{F}_i^{\text{internal}} = \sum_{i=1}^N \sum_{j \neq i} \mathbf{F}_{ij} = \mathbf{0}$$

### 1.3 Viscous Damping
A velocity-dependent drag force models immersion in a viscous fluid:
$$\mathbf{F}_i^{\text{diss}} = -\gamma m_i \mathbf{v}_i$$
The time rate of mechanical energy change $E = K + U$ satisfies:
$$\frac{dE}{dt} = -2\gamma K(t) \le 0$$
guaranteeing monotonic decay to a static configuration ($K \to 0$).

### 1.4 Boundary Conditions
Elastic specular reflection is applied at container walls $[0, L] \times [0, L]$:
- Particles crossing $x \le 0$ or $x \ge L$ have their normal velocity inverted ($v_x \leftarrow -v_x$) and position clamped to the boundary.
- Particles crossing $y \le 0$ or $y \ge L$ have $v_y \leftarrow -v_y$.

---

## 2. Numerical Integration Schemes

### 2.1 Velocity Verlet (Symplectic, Second-Order)
The system is integrated using the kick-drift-kick splitting:
$$\mathbf{v}\left(t + \frac{\Delta t}{2}\right) = \mathbf{v}(t) + \frac{1}{2} \mathbf{a}(t) \Delta t$$
$$\mathbf{r}(t + \Delta t) = \mathbf{r}(t) + \mathbf{v}\left(t + \frac{\Delta t}{2}\right) \Delta t$$
$$\text{[Apply wall boundary conditions on } \mathbf{r}(t + \Delta t) \text{ and } \mathbf{v}(t + \Delta t/2) \text{]}$$
$$\mathbf{a}(t + \Delta t) = \frac{1}{m} \mathbf{F}\left(\mathbf{r}(t + \Delta t)\right)$$
$$\mathbf{v}(t + \Delta t) = \mathbf{v}\left(t + \frac{\Delta t}{2}\right) + \frac{1}{2} \mathbf{a}(t + \Delta t) \Delta t$$

Because Velocity Verlet is symplectic, it preserves phase-space volume (Liouville's theorem), preventing secular energy drift over long integration times.

### 2.2 Forward Euler (Non-Symplectic, First-Order)
$$\mathbf{r}(t + \Delta t) = \mathbf{r}(t) + \mathbf{v}(t) \Delta t$$
$$\mathbf{v}(t + \Delta t) = \mathbf{v}(t) + \mathbf{a}(t) \Delta t$$
Euler systematically overestimates work along convex potential trajectories, introducing artificial numerical heating that scales as $\mathcal{O}(\Delta t)$.

---

## 3. Summary of Task Results

| Task | Topic | Key Quantitative Finding | Status |
| :--- | :--- | :--- | :--- |
| **Task 1** | Verification | 10/10 analytical tests passed; internal force residual $< 5.4 \times 10^{-15}\text{ N}$; $N=100\text{k}$ memory matches C `sizeof` | Verified |
| **Task 3** | Timestep Accuracy | Verlet achieves $\mathcal{O}(\Delta t^2)$ convergence ($\varepsilon_E < 0.31\%$ at $\Delta t=0.005\text{ s}$); Euler displays secular drift ($\varepsilon_E > 21\%$) | Verified |
| **Task 4** | Dissipation | Mechanical energy decays monotonically; system arrests into static equilibrium in $t_{\text{relax}} = 16.54\text{ s}$ at $\gamma = 0.50\text{ s}^{-1}$ | Verified |
| **Task 5** | Precision (float32 vs float64) | `float32` halves state memory ($1.93\times$ process savings); energy remains bounded ($\varepsilon_E \approx 3.4\%$) despite chaotic trajectory divergence | Verified |
| **Task 6** | Computational Efficiency | Compiled C kernel (`-O2`, Newton's 3rd law loop) evaluates forces in $0.25\text{ ms}$ at $N=1000$ ($261\times$ faster than Python, $88\times$ faster than NumPy) | Verified |
| **Task 7** | Memory Scalability | Dense displacement tensor $(N, N, 2)$ scales quadratically $\mathcal{O}(N^2)$; $N=50,000$ requires $37.25\text{ GiB}$ in float64, precipitating OOM failure | Verified |

---

## 4. Directory Structure & Execution Guide

```text
ProjectM/
├── core/                                         # Shared physics engine and C kernel
│   ├── physics.py                                # Vectorized physics functions
│   ├── c_kernel.py                               # Native C dynamic library loader
│   ├── pairforce.c                               # C source code
│   └── libpairforce.dylib                        # Compiled shared library
│
├── Task1_Verification/                           # Unit tests and foundational checks
│   ├── task1_verification.py
│   └── HANDOUT.md
│
├── Task3_Timestep_Accuracy/                      # Integrator comparison and convergence
│   ├── task3_accuracy.py
│   ├── results/                                  # Convergence and energy evolution plots
│   └── HANDOUT.md
│
├── Task4_Viscous_Dissipation/                    # Fluid damping study
│   ├── task4_dissipation.py
│   ├── results/                                  # Relaxation curves and trajectory plots
│   └── HANDOUT.md
│
├── Task5_Numerical_Precision/                    # Single vs double precision analysis
│   ├── task5_precision_2d.py
│   ├── results/                                  # Precision metrics, timeseries, and plots
│   └── HANDOUT.md
│
├── Task6_Computational_Efficiency/               # Benchmark: Python vs NumPy vs C
│   ├── task6_benchmark.py
│   ├── pairforce_task6.c
│   ├── results/                                  # Wall time and speedup plots
│   └── HANDOUT.md
│
├── Task7_Memory_Scalability/                     # 2D memory footprint and OOM limits
│   ├── task7_memory.py
│   ├── results/                                  # Memory measurements and scaling curves
│   └── HANDOUT.md
│
├── simulator.py                                  # Live 2D visualization & verification runner
├── AI_USAGE_LOG.md                               # Academic AI disclosure record
└── HANDOUT.md                                    # This report
```

### Execution Commands
```bash
# Run individual task experiments:
python3 Task1_Verification/task1_verification.py
python3 Task3_Timestep_Accuracy/task3_accuracy.py
python3 Task4_Viscous_Dissipation/task4_dissipation.py
python3 Task5_Numerical_Precision/task5_precision_2d.py
python3 Task6_Computational_Efficiency/task6_benchmark.py
python3 Task7_Memory_Scalability/task7_memory.py

# Launch interactive simulation:
python3 simulator.py
python3 simulator.py --backend c                  # Using compiled C backend
python3 simulator.py --preset dissipative        # Using viscous fluid preset
```

---

## 5. Academic AI Disclosure
In adherence to course policies, generative AI tools were used exclusively for programming syntax queries (Matplotlib subplot configuration, ctypes C-pointer typing, and argparse syntax). All physical formulations, mathematical proofs, and numerical evaluations were completed independently. Detailed prompt disclosures are maintained in [AI_USAGE_LOG.md](file:///Users/prakharsingh/Desktop/Study%20Material/Sem1/Scientific%20Computing/Project%201/ProjectM/AI_USAGE_LOG.md).
