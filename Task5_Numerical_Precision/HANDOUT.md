# Task 5: Numerical Precision Study (Float32 vs Float64 in 2D)

## 1. Objective
Task 5 evaluates computational trade-offs between IEEE 754 single-precision (`float32`) and double-precision (`float64`) arithmetic in molecular dynamics. We investigate:
1. Memory allocation and bandwidth reduction.
2. Long-term energy stability under lower mantissa precision.
3. Chaotic trajectory divergence quantified via relative root-mean-square position error.

---

## 2. Theoretical Analysis

### 2.1 IEEE 754 Floating-Point Formats
- **Double precision (`float64`, binary64)**:
  - 1 sign bit, 11 exponent bits, 52 explicit mantissa bits (53 bits effective).
  - Machine epsilon: $\epsilon_{\text{mach}} = 2^{-52} \approx 2.22 \times 10^{-16}$ (~15-17 significant decimal digits).
  - Storage: 8 bytes per scalar.
- **Single precision (`float32`, binary32)**:
  - 1 sign bit, 8 exponent bits, 23 explicit mantissa bits (24 bits effective).
  - Machine epsilon: $\epsilon_{\text{mach}} = 2^{-23} \approx 1.19 \times 10^{-7}$ (~6-8 significant decimal digits).
  - Storage: 4 bytes per scalar (50% reduction).

### 2.2 Lyapunov Trajectory Divergence
Nonlinear multi-particle systems with repulsive potentials and specular wall reflections are deterministic but chaotic. Microscopic round-off differences in `float32` act as perturbations $\delta \mathbf{r}(0) \sim 10^{-7}$ that amplify exponentially over time:
$$\|\delta \mathbf{r}(t)\| \sim \|\delta \mathbf{r}(0)\| e^{\lambda t}$$
where $\lambda > 0$ is the maximal Lyapunov exponent of the system.

The normalized root-mean-square position difference relative to container size $L$ is:
$$\varepsilon_{\text{pos}}(t) = \frac{1}{L} \sqrt{\frac{1}{N} \sum_{i=1}^N \|\mathbf{r}_i^{\text{float32}}(t) - \mathbf{r}_i^{\text{float64}}(t)\|^2}$$

---

## 3. Experimental Results

Execute the precision study:
```bash
python3 Task5_Numerical_Precision/task5_precision_2d.py
```

### Measured Quantitative Metrics ($N = 12$, $L = 10.0$, $\Delta t = 0.005\text{ s}$, $T = 10.0\text{ s}$)
| Metric | `float32` (Single) | `float64` (Double) | Ratio / Physical Implication |
| :--- | :--- | :--- | :--- |
| **Tracked State Memory** | 336 bytes | 672 bytes | $2.00\times$ reduction |
| **Tracked Pairwise Memory** | 1,872 bytes | 3,600 bytes | $1.92\times$ reduction |
| **Total Tracked Memory** | 2,208 bytes | 4,272 bytes | **$1.93\times$ reduction** |
| **Median Execution Time** | $0.0576\text{ s}$ | $0.0558\text{ s}$ | $1.03\times$ (scalar CPU bound) |
| **Initial Total Energy** | $79.9448\text{ J}$ | $79.9447\text{ J}$ | Identical within float32 precision |
| **Final Total Energy** | $82.7071\text{ J}$ | $80.1766\text{ J}$ | Bounded oscillation |
| **Max Relative Energy Drift** | $3.46 \times 10^{-2}$ ($3.46\%$) | $1.23 \times 10^{-2}$ ($1.23\%$) | Stable Hamiltonian conservation |
| **Final Relative Position Error** | $7.06 \times 10^{-1}$ ($70.6\%$) | Reference ($0.00$) | Chaotic trajectory divergence |

Generated plots saved in `Task5_Numerical_Precision/results/`:
- `task5_energy_comparison.png`: Energy curves showing bounded variations in both precisions.
- `task5_position_error.png`: Semilog plot of $\varepsilon_{\text{pos}}(t)$ displaying exponential growth to box-scale divergence.
- `task5_runtime_comparison.png`: Runtime comparison across 5 independent timing trials.

---

## 4. Oral Examination Technical Defenses

**Q1: Why does float32 trajectory diverge from float64 if both use the exact same physics?**
> *Defense*: The equations of motion are non-linear and sensitive to initial conditions. Single precision truncates numbers beyond the 24th bit of mantissa. This small truncation error ($\sim 10^{-7}$) acts as an infinitesimal perturbation that grows exponentially due to the positive Lyapunov exponent of hard collisions and non-linear spring interactions.

**Q2: If the trajectory diverges, why is float32 still considered useful in molecular dynamics?**
> *Defense*: In statistical physics, macro-state observables (temperature, pressure, virial stress, radial distribution functions, and energy conservation) depend on ensemble phase-space averages, not on tracking exact deterministic trajectories of individual atoms. Because Velocity Verlet bounds the energy drift even in float32 ($\approx 3.4\%$), thermodynamic ensembles remain physically valid.

**Q3: Why was float32 not significantly faster than float64 on the CPU in this benchmark?**
> *Defense*: For small particle numbers ($N=12$), execution time is dominated by Python interpreter dispatch and memory latency rather than vector arithmetic. Single-precision speedups typically manifest on SIMD vector units (AVX-512) and GPU architectures where 32-bit floating-point ALUs offer $2\times$ the arithmetic throughput of 64-bit ALUs.
