# Task 4: Viscous Dissipation & Steady-State Relaxation

## 1. Objective
Task 4 investigates mechanical energy dissipation under linear Stokes drag $\mathbf{F}_{\text{diss}} = -\gamma m \mathbf{v}$. We study:
1. The physical rate of energy loss ($dE/dt \le 0$) under varying damping coefficients $\gamma \in [0.0, 0.05, 0.10, 0.50]\text{ s}^{-1}$.
2. The relaxation time $t_{\text{relax}}$ required to arrest particles into a static potential ground state.
3. Numerical integration of damped Hamiltonian equations within the Velocity Verlet framework.

---

## 2. Theoretical Analysis

### 2.1 Dissipation Mechanics
The net force on particle $i$ combines conservative Hookean spring forces and viscous drag:
$$\mathbf{F}_i = \sum_{j \neq i} \mathbf{F}_{ij} - \gamma m_i \mathbf{v}_i$$
Taking the dot product with velocity $\mathbf{v}_i$ and summing over all $N$ particles:
$$\sum_{i=1}^N m_i \mathbf{a}_i \cdot \mathbf{v}_i = \sum_{i=1}^N \mathbf{F}_{i}^{\text{spring}} \cdot \mathbf{v}_i - \gamma \sum_{i=1}^N m_i \|\mathbf{v}_i\|^2$$
Recognizing that $\sum \mathbf{F}^{\text{spring}} \cdot \mathbf{v} = -dU/dt$ and $\sum m \mathbf{a} \cdot \mathbf{v} = dK/dt$:
$$\frac{d(K + U)}{dt} = -2\gamma K(t) \le 0$$
Because $K(t) \ge 0$ and $\gamma \ge 0$, total mechanical energy decreases monotonically ($dE/dt \le 0$) until all kinetic motion ceases ($K \to 0$).

### 2.2 Relaxation Criteria
We define the system to have reached steady-state mechanical equilibrium when:
1. **Velocity Criterion**: Mean speed $\langle \|\mathbf{v}\| \rangle < 10^{-3}\text{ m/s}$ over a 2.0-second time window.
2. **Energy Gradient Criterion**: Relative energy decay $|dE/dt| / E_0 < 10^{-3}$.

---

## 3. Experimental Results

Execute the experiment:
```bash
python3 Task4_Viscous_Dissipation/task4_dissipation.py
```

### Measured Relaxation Data ($\Delta t = 0.005\text{ s}$, $T = 20\text{ s}$, Seed 42)
| $\gamma$ ($\text{s}^{-1}$) | $E(0)$ (J) | $E_{\text{final}}$ (J) | $K_{\text{final}}$ (J) | Relaxation Time $t_{\text{relax}}$ |
| :--- | :--- | :--- | :--- | :--- |
| **0.00** | 77.1426 | 76.5603 | $7.4120 \times 10^{1}$ | Not reached (conservative) |
| **0.05** | 77.1426 | 12.1551 | $1.2087 \times 10^{1}$ | Not reached ($t_{\text{relax}} > 20\text{ s}$) |
| **0.10** | 77.1426 | 1.7677 | $1.7269 \times 10^{0}$ | Not reached ($t_{\text{relax}} > 20\text{ s}$) |
| **0.50** | 77.1426 | 0.0000 | $2.6064 \times 10^{-7}$ | **16.54 s** |

Generated plots saved in `Task4_Viscous_Dissipation/results/`:
- `task4_dissipation_KUE.png`: 4-panel subplot showing kinetic, potential, and total energy decay for each $\gamma$.
- `task4_relaxation_curves.png`: Semilog plots of $E(t)$ and $\langle \|\mathbf{v}\| \rangle(t)$ illustrating exponential decay envelopes.
- `task4_trajectories_comparison.png`: Spatial trajectory comparison showing space-filling ergodic paths at $\gamma=0$ vs localized arrest at $\gamma=0.5$.

---

## 4. Oral Examination Technical Defenses

**Q1: Why does potential energy $U$ not necessarily drop to zero as $t \to \infty$?**
> *Defense*: When particles come to rest ($K \to 0$), they arrest in local potential energy minima. If particles are compressed against walls or against neighboring particles within the cutoff radius ($r_{ij} < r_0$), static residual strain energy remains in the springs, yielding $U_{\text{final}} \ge 0$.

**Q2: How does damping affect Newton's Third Law in this system?**
> *Defense*: The internal spring forces continue to satisfy Newton's Third Law exactly ($\mathbf{F}_{ij} = -\mathbf{F}_{ji}$). However, the viscous drag force $-\gamma m \mathbf{v}$ represents momentum exchange with an external fluid reservoir, so the net external force on the system is non-zero during dissipation.

**Q3: How is viscous damping integrated in Velocity Verlet without introducing numerical instability?**
> *Defense*: Using an operator-split approach, velocity is updated using half-step accelerations combined with damping decay factors $c = \exp(-\gamma \Delta t / 2) \approx 1 - \gamma \Delta t / 2$, preserving second-order accuracy and preventing numerical divergence even at high damping coefficients.
