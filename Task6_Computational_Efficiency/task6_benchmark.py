"""Task 6: fair 2D pair-force benchmark (Python loop vs NumPy vs submitted C).

Uses the force law and parameters from the group's uploaded task3_4.py:
    k=15.0, r0=1.5, rc=1.5, positions in a 2D square, float64.

For each N in [50, 200, 500, 1000], one deterministic position array is
shared by all methods. Each method is timed in at least five independent
samples. Each sample is a calibrated batch; reported time is per force call.
The C ABI preserves the newly uploaded pairforce.c argument order:
    compute_pair_forces(N, pos, F, rc, r0, k).

Run:
    python task6_benchmark.py
    python task6_benchmark.py --samples 7
    python task6_benchmark.py --sizes 50 200 500 1000 --samples 5

GCC (MSYS2/MinGW-w64 on Windows) must be available on PATH.
"""
from __future__ import annotations

import argparse
import ctypes
import csv
import gc
import math
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from statistics import median
from time import perf_counter

# Ensure project root is in sys.path so task3_4 can be imported from any working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from core.physics import pair_forces
except ImportError:
    from task3_4 import pair_forces

# Match task3_4.py exactly.
N_REFERENCE = 12
L_REFERENCE = 10.0
MARGIN = 0.8
K_SPRING = 15.0
R0 = 1.5
RC = 1.5
EPS = 1.0e-12
SEED = 42
DEFAULT_SIZES = [50, 200, 500, 1000]
DEFAULT_SAMPLES = 5
TARGET_BATCH_SECONDS = 0.025
MAX_BATCH_REPEATS = 2000

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "task6_results"
C_SOURCE = SCRIPT_DIR / "pairforce_task6.c"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_c_library() -> Path:
    compiler = shutil.which("gcc") or (shutil.which("clang") if platform.system() == "Darwin" else None)
    if compiler is None:
        raise RuntimeError(
            "GCC was not found. On Windows install MSYS2/MinGW-w64, add its mingw64\\bin "
            "folder to PATH, open a new VS Code terminal, and check: gcc --version"
        )
    if not C_SOURCE.exists():
        raise FileNotFoundError(f"Missing C source: {C_SOURCE}")

    system = platform.system()
    if system == "Windows":
        library_path = SCRIPT_DIR / "pairforce.dll"
        command = [compiler, "-O2", "-shared", "-o", str(library_path), str(C_SOURCE), "-lm"]
    elif system == "Darwin":
        library_path = SCRIPT_DIR / "libpairforce.dylib"
        command = [compiler, "-O2", "-dynamiclib", "-fPIC", "-o", str(library_path), str(C_SOURCE), "-lm"]
    else:
        library_path = SCRIPT_DIR / "libpairforce.so"
        command = [compiler, "-O2", "-shared", "-fPIC", "-o", str(library_path), str(C_SOURCE), "-lm"]

    proc = subprocess.run(command, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "C compilation failed.\nCommand: " + " ".join(command) +
            "\nstdout:\n" + proc.stdout + "\nstderr:\n" + proc.stderr
        )
    return library_path


def load_c_function(library_path: Path):
    library = ctypes.CDLL(str(library_path))
    function = library.compute_pair_forces
    double_ptr = ctypes.POINTER(ctypes.c_double)
    # IMPORTANT: order follows uploaded pairforce.c: N, pos, F, rc, r0, k.
    function.argtypes = [
        ctypes.c_int,
        double_ptr,
        double_ptr,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
    ]
    function.restype = None
    return function, library


def make_positions(n: int) -> tuple[np.ndarray, float]:
    """Generate one reproducible dataset and approximately preserve density.

    The reference state samples the accessible square [0.8, 9.2]^2 for N=12.
    Scale that accessible side by sqrt(N/12), then restore the 0.8 boundary
    margin. Every implementation receives this exact same C-contiguous array.
    """
    accessible_side = (L_REFERENCE - 2.0 * MARGIN) * math.sqrt(n / N_REFERENCE)
    box_length = accessible_side + 2.0 * MARGIN
    rng = np.random.default_rng(SEED + n)
    pos = rng.uniform(MARGIN, box_length - MARGIN, size=(n, 2))
    return np.ascontiguousarray(pos, dtype=np.float64), box_length


def python_double_loop(pos: np.ndarray) -> np.ndarray:
    """Pure Python unique-pair calculation; force accumulation uses Newton III."""
    n = pos.shape[0]
    forces = np.zeros((n, 2), dtype=np.float64)
    for i in range(n):
        xi = float(pos[i, 0])
        yi = float(pos[i, 1])
        for j in range(i + 1, n):
            dx = xi - float(pos[j, 0])
            dy = yi - float(pos[j, 1])
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > EPS and dist <= RC:
                factor = -K_SPRING * (dist - R0) / dist
                fx = factor * dx
                fy = factor * dy
                forces[i, 0] += fx
                forces[i, 1] += fy
                forces[j, 0] -= fx
                forces[j, 1] -= fy
    return forces


def numpy_vectorized(pos: np.ndarray) -> np.ndarray:
    """Vectorized pair forces imported from task3_4.py."""
    return pair_forces(pos, k=K_SPRING, r0=R0, rc=RC)


def make_c_wrapper(c_function):
    def compiled_c(pos: np.ndarray) -> np.ndarray:
        pos = np.ascontiguousarray(pos, dtype=np.float64)
        forces = np.empty_like(pos)
        ptr = ctypes.POINTER(ctypes.c_double)
        c_function(
            ctypes.c_int(pos.shape[0]),
            pos.ctypes.data_as(ptr),
            forces.ctypes.data_as(ptr),
            ctypes.c_double(RC),
            ctypes.c_double(R0),
            ctypes.c_double(K_SPRING),
        )
        return forces
    return compiled_c


def calibrate_repeats(function, pos: np.ndarray) -> int:
    start = perf_counter()
    function(pos)
    elapsed = max(perf_counter() - start, 1.0e-9)
    return max(1, min(MAX_BATCH_REPEATS, int(math.ceil(TARGET_BATCH_SECONDS / elapsed))))


def timed_sample(function, pos: np.ndarray, repeats: int) -> float:
    start = perf_counter()
    for _ in range(repeats):
        function(pos)
    return (perf_counter() - start) / repeats


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_benchmark(sizes: list[int], samples: int) -> None:
    library_path = build_c_library()
    c_function, library_handle = load_c_function(library_path)
    c_loop = make_c_wrapper(c_function)
    methods = {
        "Python double loop": python_double_loop,
        "NumPy vectorized": numpy_vectorized,
        "Compiled C (-O2)": c_loop,
    }
    names = list(methods)
    summary_rows, sample_rows, verify_rows = [], [], []

    print("TASK 6 - 2D PAIR-FORCE EFFICIENCY BENCHMARK")
    print(f"OS: {platform.platform()}")
    print(f"Python: {sys.version.split()[0]} | NumPy: {np.__version__}")
    print(f"C library: {library_path.name} | compiler: {shutil.which('gcc') or shutil.which('clang')}")
    print(f"Force settings: k={K_SPRING}, r0={R0}, rc={RC}, EPS={EPS}")
    print(f"N values: {sizes} | timing samples: {samples} | C arg order: N,pos,F,rc,r0,k\n")

    for n in sizes:
        pos, box_length = make_positions(n)
        outputs = {name: fn(pos) for name, fn in methods.items()}
        f_ref = outputs["Python double loop"]
        np_err = float(np.max(np.abs(f_ref - outputs["NumPy vectorized"])))
        c_err = float(np.max(np.abs(f_ref - outputs["Compiled C (-O2)"])))
        net_force = float(np.linalg.norm(np.sum(f_ref, axis=0)))
        scale = max(float(np.linalg.norm(f_ref)), 1.0)
        if not np.allclose(f_ref, outputs["NumPy vectorized"], rtol=1e-8, atol=1e-8):
            raise AssertionError(f"NumPy mismatch at N={n}: {np_err}")
        if not np.allclose(f_ref, outputs["Compiled C (-O2)"], rtol=1e-8, atol=1e-8):
            raise AssertionError(f"C mismatch at N={n}: {c_err}")
        verify_rows.append({
            "N": n,
            "box_length": f"{box_length:.9f}",
            "max_abs_diff_python_numpy": f"{np_err:.12e}",
            "max_abs_diff_python_c": f"{c_err:.12e}",
            "net_force_norm_python": f"{net_force:.12e}",
            "net_force_relative": f"{net_force / scale:.12e}",
            "all_methods_match": "PASS",
        })

        batch_counts = {name: calibrate_repeats(fn, pos) for name, fn in methods.items()}
        times = {name: [] for name in names}
        gc_enabled = gc.isenabled()
        gc.disable()
        try:
            for s in range(samples):
                offset = s % len(names)
                order = names[offset:] + names[:offset]
                for name in order:
                    elapsed = timed_sample(methods[name], pos, batch_counts[name])
                    times[name].append(elapsed)
                    sample_rows.append({
                        "N": n,
                        "box_length": f"{box_length:.9f}",
                        "method": name,
                        "sample": s + 1,
                        "calls_in_sample": batch_counts[name],
                        "time_per_call_seconds": f"{elapsed:.12e}",
                        "time_per_call_milliseconds": f"{elapsed * 1000:.9f}",
                    })
        finally:
            if gc_enabled:
                gc.enable()

        baseline = median(times["Python double loop"])
        for name in names:
            vals = times[name]
            med = median(vals)
            speedup = baseline / med
            summary_rows.append({
                "N": n,
                "box_length": f"{box_length:.9f}",
                "method": name,
                "median_seconds": f"{med:.12e}",
                "median_milliseconds": f"{med * 1000:.9f}",
                "min_milliseconds": f"{min(vals) * 1000:.9f}",
                "max_milliseconds": f"{max(vals) * 1000:.9f}",
                "speedup_vs_python": f"{speedup:.6f}",
                "timing_samples": samples,
                "calls_per_sample": batch_counts[name],
            })
        print(f"N={n:4d} | max |Python-NumPy|={np_err:.3e} | max |Python-C|={c_err:.3e}")
        for name in names:
            print(f"    {name:<22} median {median(times[name])*1000:10.5f} ms | speedup {baseline/median(times[name]):9.3f}x")

    write_csv(OUT_DIR / "task6_timing_summary.csv",
              ["N", "box_length", "method", "median_seconds", "median_milliseconds", "min_milliseconds", "max_milliseconds", "speedup_vs_python", "timing_samples", "calls_per_sample"], summary_rows)
    write_csv(OUT_DIR / "task6_timing_samples.csv",
              ["N", "box_length", "method", "sample", "calls_in_sample", "time_per_call_seconds", "time_per_call_milliseconds"], sample_rows)
    write_csv(OUT_DIR / "task6_verification.csv",
              ["N", "box_length", "max_abs_diff_python_numpy", "max_abs_diff_python_c", "net_force_norm_python", "net_force_relative", "all_methods_match"], verify_rows)

    # Wall-time plot (log scale as required).
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    for name in names:
        subset = [r for r in summary_rows if r["method"] == name]
        ax.plot([int(r["N"]) for r in subset], [float(r["median_milliseconds"]) for r in subset], marker="o", linewidth=2, label=name)
    ax.set_yscale("log")
    ax.set_xlabel("Number of particles, N")
    ax.set_ylabel("Median wall-clock time per force call (ms, log scale)")
    ax.set_title("Task 6: Pair-force execution time vs system size")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "task6_wall_time_vs_N.png", dpi=220)
    plt.close(fig)

    # Speedup plot.
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    for name in names:
        subset = [r for r in summary_rows if r["method"] == name]
        ax.plot([int(r["N"]) for r in subset], [float(r["speedup_vs_python"]) for r in subset], marker="o", linewidth=2, label=name)
    ax.axhline(1.0, linestyle="--", linewidth=1, label="Python baseline = 1x")
    ax.set_xlabel("Number of particles, N")
    ax.set_ylabel("Speedup relative to Python double loop")
    ax.set_title("Task 6: Speedup vs Python baseline")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "task6_speedup_vs_N.png", dpi=220)
    plt.close(fig)

    try:
        compiler_cmd = shutil.which("gcc") or shutil.which("clang")
        compiler_line = subprocess.run([compiler_cmd, "--version"], capture_output=True, text=True).stdout.splitlines()[0] if compiler_cmd else "Not detected"
    except Exception:
        compiler_line = "Not detected"

    env = [
        "TASK 6 BENCHMARK ENVIRONMENT",
        f"Timestamp: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"OS: {platform.platform()}",
        f"Processor: {platform.processor() or 'Not reported'}",
        f"Python: {sys.version.replace(os.linesep, ' ')}",
        f"NumPy: {np.__version__}",
        f"Compiler: {compiler_line}",
        "Compiler flags: -O2 shared library",
        f"N sizes: {sizes}",
        f"Samples per method per N: {samples}",
        f"Position density rule: accessible_side=(10.0-1.6)*sqrt(N/12); margin=0.8",
        f"Force parameters: k={K_SPRING}, r0={R0}, rc={RC}, epsilon={EPS}",
        "C function ABI: compute_pair_forces(int N, const double *pos, double *F, double rc, double r0, double k)",
        "Timing includes Python-to-ctypes function-call overhead for the C method; excludes compilation and data generation.",
    ]
    (OUT_DIR / "task6_environment.txt").write_text("\n".join(env) + "\n", encoding="utf-8")

    print(f"\nSaved Task 6 files in: {OUT_DIR}")
    print("  task6_timing_summary.csv")
    print("  task6_timing_samples.csv")
    print("  task6_verification.csv")
    print("  task6_wall_time_vs_N.png")
    print("  task6_speedup_vs_N.png")
    print("  task6_environment.txt")
    _ = library_handle


def run_task_6(sizes: list[int] | None = None, samples: int = DEFAULT_SAMPLES) -> None:
    if sizes is None:
        sizes = DEFAULT_SIZES
    if any(n < 2 for n in sizes):
        raise ValueError("All N values must be >= 2.")
    if samples < 5:
        raise ValueError("Task 6 requires at least 5 timing samples; use samples >= 5.")
    run_benchmark(sizes, samples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Task 6: 2D Python/NumPy/C pair-force benchmark.")
    parser.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    args = parser.parse_args()
    run_task_6(args.sizes, args.samples)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        run_task_6()
