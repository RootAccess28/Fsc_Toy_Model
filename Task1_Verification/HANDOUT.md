# Task 1: System Setup, Memory Strides & Physical Verification

## 1. Objective
Task 1 establishes the mathematical formulation and numerical verification of the 2D molecular toy model. It validates:
- Memory alignment and stride arithmetic of contiguous 2D coordinate arrays against C pointer mechanics.
- Metric properties of vectorized Euclidean distance matrices (symmetry, zero diagonal, self-exclusion).
- Strict conservation of linear momentum under Newton's Third Law ($\sum \mathbf{F}_i = \mathbf{0}$).
- Elastic boundary containment at container walls $[0, L] \times [0, L]$.

---

## 2. Theoretical Formulation & Array Mechanics

### 2.1 Memory Strides and C Representation
In both C and NumPy, a 2D array of coordinates has shape $(N, 2)$ stored in row-major order:
- **Double precision (`float64`, 8 bytes)**:
  - `itemsize = 8` bytes.
  - `strides = (16, 8)`: each row advance requires 16 bytes ($2 \times 8$); each coordinate step requires 8 bytes.
- **Single precision (`float32`, 4 bytes)**:
  - `itemsize = 4` bytes.
  - `strides = (8, 4)`: each row advance requires 8 bytes ($2 \times 4$); each coordinate step requires 4 bytes.
- **Memory Footprint for $N = 100,000$ particles**:
  $$M_{\text{pos}} = N \times 2 \times \text{sizeof(type)}$$
  - `float64`: $100,000 \times 2 \times 8 = 1,600,000\text{ bytes} \approx 1.526\text{ MiB}$ ($1.60\text{ MB}$ decimal).
  - `float32`: $100,000 \times 2 \times 4 = 800,000\text{ bytes} \approx 0.763\text{ MiB}$ ($0.80\text{ MB}$ decimal).
  - This matches C's `sizeof(double)` and `sizeof(float)` verified via ctypes.

### 2.2 Vectorized Pairwise Displacements
To avoid interpreted loop overhead, pairwise displacements are evaluated using array broadcasting:
$$\mathbf{d}_{ij} = \mathbf{r}_i - \mathbf{r}_j \in \mathbb{R}^{N \times N \times 2}$$
The Euclidean distance matrix is:
$$r_{ij} = \|\mathbf{d}_{ij}\| = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}$$

- **Symmetry**: Distance is symmetric ($r_{ij} = r_{ji}$), yielding residual $\|D - D^T\|_\infty = 0$.
- **Diagonal**: Self-distance is strictly zero ($r_{ii} = 0$).

### 2.3 Interaction Cutoff & Neighbor Mask
To truncate non-interacting pairs beyond range $r_c$, we define:
$$\text{neighbor}_{ij} = (0 < r_{ij} \le r_c)$$
The $r_{ij} > 0$ criterion strictly enforces self-interaction exclusion ($i \neq j$).

### 2.4 Newton's Third Law & Momentum Conservation
Spring forces between pairs are equal and opposite:
$$\mathbf{F}_{ij} = -k (r_{ij} - r_0) \hat{\mathbf{r}}_{ij} = -\mathbf{F}_{ji}$$
Summing over all particles:
$$\sum_{i=1}^N \mathbf{F}_i = \sum_{i=1}^N \sum_{j \neq i} \mathbf{F}_{ij} = \mathbf{0}$$
In double precision, the evaluated net force residual is $\| \sum \mathbf{F} \| < 5.4 \times 10^{-15}\text{ N}$, within machine epsilon.

### 2.5 Specular Wall Reflection
When a particle reaches a boundary of $[0, L] \times [0, L]$:
- Normal velocity component is inverted: $v_{\perp} \leftarrow -v_{\perp}$.
- Position is clamped to the boundary: $x = 0$ or $x = L$.

---

## 3. Verification Test Results

Execute the verification suite:
```bash
python3 Task1_Verification/task1_verification.py
```

### Measured Execution Output
```text
Task 1: Numerical Verification Suite
--------------------------------------------------
  1. Memory layout (float64: 1.60 MB, float32: 0.80 MB): OK
  2. Distance matrix symmetry (residual = 0.00e+00): OK
  3. Zero diagonal entries: OK
  4. Self-neighbor exclusion: OK
  5. 4-particle Pythagorean benchmark: OK
  6. 4-particle neighbor counts (rc=2.1): OK
  7. Cutoff scaling monotonicity (rc in [1.0, 1.5, 2.0, 3.0]): OK
  8. Newton's Third Law (net force = 5.40e-15 N): OK
  9. Wall reflection and clamping: OK
 10. Initial energy consistency (E0 = 77.1426 J): OK
--------------------------------------------------
Result: 10/10 tests passed.
--------------------------------------------------
```

---

## 4. Oral Examination Technical Defenses

**Q1: Explain what `pos.strides` represents physically in memory.**
> *Defense*: Strides indicate the number of bytes that the memory pointer must advance to step to the next element along each dimension. For an $(N, 2)$ float64 array, the strides are `(16, 8)`. Stepping to the next coordinate of the same particle advances 8 bytes (one double); stepping to the next particle advances 16 bytes (two doubles).

**Q2: How does the vectorized pairwise distance formulation avoid Python loop overhead?**
> *Defense*: Expanding `pos[:, None, :]` shape $(N, 1, 2)$ and subtracting `pos[None, :, :]` shape $(1, N, 2)$ performs the displacement calculation entirely within compiled C loops internal to NumPy, eliminating bytecode interpretation and boxing overhead.

**Q3: Why must self-interaction ($i=j$) be explicitly excluded from the neighbor mask?**
> *Defense*: At $i=j$, the distance $r_{ii} = 0$. Evaluating the unit vector $\hat{\mathbf{r}}_{ii} = \mathbf{d}_{ii} / r_{ii}$ introduces a $0/0$ division singularity. Filtering with $0 < r_{ij} \le r_c$ guarantees no NaN values and ensures particles do not exert non-physical self-forces.
