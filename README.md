# Toy Model: Exploring Molecular & Atomic Interactions

A 2D particle simulation modeling atomic interactions inside a bounded box using classical mechanics, custom force computations, wall reflections, and energy tracking[cite: 1]. This repository includes Python, NumPy-vectorized, and compiled C implementations with comparative benchmarking[cite: 1].

---

## Project Structure

```text
.
├── README.md
├── requirements.txt
├── src/
│   ├── simulation.py      # Core simulation: positions, reflections, forces, and steppers
│   └── pairforce.c        # Compiled C baseline for pairwise force calculations
├── benchmark.py           # Benchmarking harness (Python loop vs. C vs. NumPy)
└── notebook.ipynb         # Interactive walkthrough, derivations, and plots
```

---

## Setup & Installation

### 1. Prerequisites
- Python 3.9+
- GCC / Clang C compiler (`gcc` with `-lm` support)[cite: 1]
- Git

### 2. Python Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Compile the C Benchmark Engine
Compile the C implementation with `-O2` optimization and link the math library[cite: 1]:
```bash
gcc -O2 src/pairforce.c -o src/pairforce -lm
```

---

## Module Overview

1. **Memory Layout & Precision Analysis (`Section 1`)**  
   - Inspects `dtype`, `itemsize`, `nbytes`, and `strides` for particle arrays[cite: 1].
   - Compares row-major memory addressing in NumPy against pointer arithmetic and `sizeof(double)` / `sizeof(float)` in C[cite: 1].
   - Explores memory consumption and floating-point accumulation for $N = 100,000$ particles[cite: 1].

2. **Boundary Conditions (`Section 2`)**  
   - `walls_reflect(pos, vel, L)`: Clamps out-of-bound particles back to $[0, L]$ and flips the normal velocity component[cite: 1].

3. **Neighbor Search via Broadcasting (`Section 3`)**  
   - `pairwise_distances(pos, rc)`: Computes pairwise displacements $(N, N, 2)$, Euclidean distances $(N, N)$, and boolean neighbor masks without Python loops[cite: 1].

4. **Pairwise Forces & Benchmarking (`Section 4`)**  
   - Implements Hooke's-style pairwise interaction forces within cutoff $r_c$[cite: 1]:  
     $$F_{ij} = -k (r - r_0) \frac{\mathbf{r}}{\|\mathbf{r}\|}$$
   - Evaluates three approaches across $N \in \{50, 200, 500, 1000\}$[cite: 1]:
     1. Pure-Python double loop[cite: 1].
     2. Compiled C executable (`pairforce.c`)[cite: 1].
     3. Fully vectorized NumPy broadcasting[cite: 1].
   - Generates wall-time vs. $N$ plots on a logarithmic scale[cite: 1].

5. **Kinematic Integrators & Energy Conservation (`Sections 5–7`)**  
   - `step_once`: Standard Euler update[cite: 1].
   - `step_smooth`: Velocity-Verlet update splitting force steps across half time intervals ($v_{\text{half}}$, $x_{\text{new}}$, $v_{\text{new}}$) to stabilize kinetic energy[cite: 1].
   - Kinetic energy diagnostics ($E_k = \frac{1}{2}\sum m v^2$) across simulation runtime[cite: 1].

---

## Running the Code

* **Run the benchmark comparison:**
  ```bash
  python benchmark.py
  ```
* **Run interactive simulation & plots:**
  ```bash
  jupyter notebook notebook.ipynb
  ```

---

## Assignment Checklist

- [ ] Particles stay bounded inside the box of side length $L$[cite: 1].
- [ ] Forces satisfy Newton's third law (equal and opposite: $\sum F \approx 0$)[cite: 1].
- [ ] Kinetic energy remains steady using `step_smooth`[cite: 1].
- [ ] Verification of time-step stability under smaller $\Delta t$[cite: 1].
- [ ] Memory calculations for $N=100,000$ validated both analytically and via NumPy introspection[cite: 1].
- [ ] `pairwise_distances` operates fully vectorized with zero loops[cite: 1].
- [ ] C program `pairforce.c` compiled with `gcc -O2 -lm`[cite: 1].
- [ ] Wall-time vs. $N$ log-scale benchmark plot generated comparing all three force implementations[cite: 1].            
