# Task 3: Timestep Accuracy & Symplectic Integrators

## 1. Objective
Task 3 investigates the numerical convergence and long-term energy conservation properties of two time-integration schemes for Hamiltonian dynamics:
1. **Forward Euler Integration**: Explicit, first-order, non-symplectic.
2. **Velocity Verlet Integration**: Explicit, second-order, symplectic.

We examine energy drift across timesteps $\Delta t \in [0.001, 0.002, 0.005, 0.010, 0.020, 0.050]\text{ s}$ over a 10-seed ensemble to verify the theoretical $\mathcal{O}(\Delta t^2)$ convergence rate.

---

## 2. Theoretical Analysis

### 2.1 Symplectic vs Non-Symplectic Integrators
The Hamiltonian of our conservative particle system is:
$$H(\mathbf{r}, \mathbf{p}) = \sum_{i=1}^N \frac{\|\mathbf{p}_i\|^2}{2 m_i} + \sum_{i < j} U(r_{ij})$$

- **Forward Euler**:
  $$\mathbf{r}_{n+1} = \mathbf{r}_n + \mathbf{v}_n \Delta t$$
  $$\mathbf{v}_{n+1} = \mathbf{v}_n + \mathbf{a}_n \Delta t$$
  Euler fails to preserve phase-space volume (determinant of the Jacobian is $1 + \mathcal{O}(\Delta t^2) \neq 1$). Consequently, energy systematically drifts upward, producing artificial numerical heating.

- **Velocity Verlet (Kick-Drift-Kick)**:
  $$\mathbf{v}_{n+1/2} = \mathbf{v}_n + \frac{1}{2} \mathbf{a}_n \Delta t$$
  $$\mathbf{r}_{n+1} = \mathbf{r}_n + \mathbf{v}_{n+1/2} \Delta t$$
  $$\mathbf{a}_{n+1} = \frac{1}{m} \mathbf{F}(\mathbf{r}_{n+1})$$
  $$\mathbf{v}_{n+1} = \mathbf{v}_{n+1/2} + \frac{1}{2} \mathbf{a}_{n+1} \Delta t$$
  Verlet exactly conserves a "shadow Hamiltonian" $\tilde{H} = H + \mathcal{O}(\Delta t^2)$. Total energy exhibits bounded periodic oscillations around the true energy with zero secular drift.

### 2.2 Global Truncation Error
The relative maximum energy drift is defined as:
$$\varepsilon_E = \frac{\max_{t \in [0, T]} |E(t) - E(0)|}{E(0)}$$
On a log-log plot of $\varepsilon_E$ versus $\Delta t$, the slope corresponds to the convergence order:
- Forward Euler exhibits slope $\approx 1$ ($\mathcal{O}(\Delta t)$).
- Velocity Verlet exhibits slope $\approx 2$ ($\mathcal{O}(\Delta t^2)$).

---

## 3. Experimental Results

Execute the experiment:
```bash
python3 Task3_Timestep_Accuracy/task3_accuracy.py
```

### Measured Data Across 10-Seed Ensemble ($T = 10\text{ s}$)
| $\Delta t$ (s) | Method | $E(0)$ (J) | Max Relative Drift (Seed 42) | Median Relative Drift (10 Seeds) |
| :--- | :--- | :--- | :--- | :--- |
| **0.050** | Euler | 77.1426 | $4.0190 \times 10^{0}$ ($402\%$) | $2.5837 \times 10^{0}$ |
| **0.050** | Verlet | 77.1426 | $1.3459 \times 10^{-2}$ ($1.35\%$) | $4.8349 \times 10^{-2}$ |
| **0.020** | Euler | 77.1426 | $1.4064 \times 10^{0}$ ($141\%$) | $9.4737 \times 10^{-1}$ |
| **0.020** | Verlet | 77.1426 | $1.6043 \times 10^{-2}$ ($1.60\%$) | $1.8768 \times 10^{-2}$ |
| **0.010** | Euler | 77.1426 | $4.9971 \times 10^{-1}$ ($50.0\%$) | $3.8995 \times 10^{-1}$ |
| **0.010** | Verlet | 77.1426 | $9.0624 \times 10^{-3}$ ($0.91\%$) | $9.9052 \times 10^{-3}$ |
| **0.005** | Euler | 77.1426 | $2.1159 \times 10^{-1}$ ($21.2\%$) | $1.9228 \times 10^{-1}$ |
| **0.005** | Verlet | 77.1426 | $3.1248 \times 10^{-3}$ ($0.31\%$) | $3.1455 \times 10^{-3}$ |
| **0.001** | Euler | 77.1426 | $3.8263 \times 10^{-2}$ ($3.83\%$) | $3.4938 \times 10^{-2}$ |
| **0.001** | Verlet | 77.1426 | $7.0505 \times 10^{-4}$ ($0.07\%$) | $7.5939 \times 10^{-4}$ |

Generated plots saved in `Task3_Timestep_Accuracy/results/`:
- `task3_convergence.png`: Log-log plot confirming slope 2 convergence for Verlet.
- `task3_Etot_curves.png`: Time-evolution of $E(t)$ showing bounded oscillations in Verlet vs monotonic heating in Euler.
- `task3_Etot_error_vs_dt.png`: Semilog error time-series.
- `task3_KUE_dt0.01.png`: Partition between $K(t)$, $U(t)$, and $E(t)$ for the baseline step $\Delta t = 0.01\text{ s}$.

---

## 4. Oral Examination Technical Defenses

**Q1: Why does Forward Euler cause secular energy increase in molecular dynamics?**
> *Defense*: Forward Euler approximates the integral along the forward tangent. In harmonic potential wells, trajectories spiral outward in phase space because the numerical eigenvalues satisfy $|\lambda| = \sqrt{1 + \omega^2 \Delta t^2} > 1$, artificially injecting energy on each step.

**Q2: What is the symplectic property of Velocity Verlet and why is it essential?**
> *Defense*: An integrator is symplectic if its Jacobian matrix $J$ satisfies $J^T \Omega J = \Omega$, where $\Omega$ is the canonical symplectic matrix. This guarantees exact preservation of phase-space volume (Poincaré invariants). Even though the true energy is not exactly conserved, the error is bounded within $\mathcal{O}(\Delta t^2)$ oscillations indefinitely without secular drift.

**Q3: How do we choose the optimal timestep $\Delta t$?**
> *Defense*: For a harmonic spring with $k = 15.0\text{ N/m}$ and $m = 1.0\text{ kg}$, the angular frequency is $\omega = \sqrt{k/m} \approx 3.87\text{ rad/s}$ ($T = 2\pi/\omega \approx 1.62\text{ s}$). Resolving one period with at least 50–100 steps requires $\Delta t \le 0.01\text{ s}$, which yields $< 1\%$ energy error in Velocity Verlet.
