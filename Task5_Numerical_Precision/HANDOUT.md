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
| **Tracked State Memory** | 240 bytes | 480 bytes | $2.00\times$ reduction ($50.0\%$ savings) |
| **Tracked Pairwise Memory** | 1,872 bytes | 3,600 bytes | $1.92\times$ reduction ($48.0\%$ savings) |
| **Total Tracked Memory** | 2,112 bytes | 4,080 bytes | **$1.93\times$ reduction ($48.2\%$ savings)** |
| **Median Execution Time** | $0.0587\text{ s}$ | $0.0571\text{ s}$ | `float32` is $2.8\%$ slower (CPU casting overhead) |
| **Initial Total Energy** | $77.142555\text{ J}$ | $77.142551\text{ J}$ | Exact match to $6$ significant figures |
| **Final Total Energy** | $76.910179\text{ J}$ | $76.902022\text{ J}$ | Bounded oscillation |
| **Max Relative Energy Drift** | $3.029 \times 10^{-3}$ ($0.303\%$) | $3.125 \times 10^{-3}$ ($0.312\%$) | Integrator truncation error dominates |
| **Final Relative Position Error** | $3.903 \times 10^{-1}$ ($39.0\%$) | Reference ($0.00$) | Chaotic Lyapunov trajectory divergence |

### 3.1 Critical Physical Interpretation
1. **Energy Conservation vs Machine Precision**:
   Notice that `float32` exhibits a maximum energy drift of $0.003029$, which is marginally *lower* than `float64` ($0.003125$). **We do not claim that `float64` automatically yields superior energy conservation in this experiment.** Because Velocity Verlet is symplectic, it exactly conserves a nearby shadow Hamiltonian. At $\Delta t = 0.005\text{ s}$, the discretization truncation error scales as $\mathcal{O}(\Delta t^2) \approx 2.5 \times 10^{-5}$, which is roughly two orders of magnitude larger than single-precision round-off ($\epsilon_{\text{mach}} \approx 1.19 \times 10^{-7}$). Discretization error completely dominates machine round-off, causing both precisions to display nearly identical energy drift.
2. **Execution Runtime on Modern 64-bit Hardware**:
   `float32` is approximately $2.8\%$ slower ($0.0587\text{ s}$ vs $0.0571\text{ s}$). For small systems ($N=12$), the array size ($2.1\text{ KB}$) fits entirely within the CPU's ultra-fast L1 cache ($128\text{ KB}$), so memory bandwidth is not saturated. Instead, Python's native 64-bit float representation introduces minor scalar type-casting overhead when interacting with NumPy `float32` arrays. Single-precision speedups only emerge when memory bandwidth is saturated or when utilizing SIMD/GPU hardware.
3. **Phase-Space Trajectory Divergence**:
   Despite identical energy conservation, individual particle trajectories diverge exponentially. Single-precision truncation error acts as an infinitesimal perturbation ($\sim 10^{-7}$) that is amplified by the positive Lyapunov exponent of particle-particle collisions and wall reflections, reaching $39.0\%$ container-scale separation by $t = 10\text{ s}$.

Generated plots saved in `Task5_Numerical_Precision/results/`:
- `task5_energy_comparison.png`: Energy curves showing bounded variations in both precisions.
- `task5_position_error.png`: Semilog plot of $\varepsilon_{\text{pos}}(t)$ displaying exponential growth to box-scale divergence.
- `task5_runtime_comparison.png`: Runtime comparison across 5 independent timing trials.

---

## 4. Oral Examination Technical Defenses

**Q1: Why does float32 trajectory diverge from float64 if both use the exact same physics?**
> *Defense*: The equations of motion are non-linear and sensitive to initial conditions. Single precision truncates numbers beyond the 24th bit of mantissa. This small truncation error ($\sim 10^{-7}$) acts as an infinitesimal perturbation that grows exponentially due to the positive Lyapunov exponent of hard collisions and non-linear spring interactions.

**Q2: If the trajectory diverges, why is float32 still considered useful in molecular dynamics?**
> *Defense*: In statistical physics, macroscopic observables (temperature, pressure, virial stress, radial distribution functions, and energy conservation) depend on ensemble phase-space averages, not on tracking exact deterministic trajectories of individual atoms. Because Velocity Verlet bounds the energy drift even in float32 ($\approx 0.30\%$), thermodynamic ensembles remain physically valid.

**Q3: Why was float32 not significantly faster than float64 on the CPU in this benchmark?**
> *Defense*: For small particle numbers ($N=12$), total data footprint is under $3\text{ KB}$, residing entirely within L1 CPU cache. Execution time is bounded by Python interpreter overhead and 64-bit CPU register conversion rather than memory bandwidth. Single-precision speedups emerge when arrays exceed CPU cache capacity or when utilizing hardware SIMD/GPU vector units.
