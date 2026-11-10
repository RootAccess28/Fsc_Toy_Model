"""
Compiled C Pair Force Acceleration Module
=========================================
Dynamically builds and binds pairforce.c via ctypes for ~30x-88x speedup.
"""

import ctypes
import platform
import shutil
import subprocess
from pathlib import Path
import numpy as np

_C_PAIR_FORCES_FN = None


def get_c_pair_forces_fn():
    """
    Compiles and loads the native C pair force kernel (pairforce.c) via ctypes.
    Returns a callable pair_forces_c(pos, k=4000.0, r0=0.70, rc=0.70) with ~30x-88x speedup.
    """
    global _C_PAIR_FORCES_FN
    if _C_PAIR_FORCES_FN is not None:
        return _C_PAIR_FORCES_FN

    core_dir = Path(__file__).resolve().parent
    root_dir = core_dir.parent

    # Candidate source locations
    candidates = [
        core_dir / "pairforce.c",
        root_dir / "pairforce.c",
        root_dir / "Task6_Computational_Efficiency" / "pairforce_task6.c",
        root_dir / "Task6_Revised_2D" / "pairforce_task6.c",
    ]
    c_source = None
    for cand in candidates:
        if cand.exists():
            c_source = cand
            break

    if c_source is None:
        raise FileNotFoundError("Could not find pairforce.c in core/, project root, or Task6/.")

    system = platform.system()
    lib_name = (
        "pairforce.dll"
        if system == "Windows"
        else ("libpairforce.dylib" if system == "Darwin" else "libpairforce.so")
    )
    lib_path = core_dir / lib_name

    if not lib_path.exists():
        compiler = "clang" if shutil.which("clang") and system == "Darwin" else "gcc"
        if not shutil.which(compiler):
            compiler = shutil.which("gcc") or shutil.which("clang")
        if not compiler:
            raise RuntimeError("No C compiler (gcc/clang) found on PATH to build pairforce.c.")

        flags = ["-O2", "-shared"] if system != "Darwin" else ["-O2", "-dynamiclib", "-fPIC"]
        cmd = [compiler] + flags + ["-o", str(lib_path), str(c_source), "-lm"]
        subprocess.run(cmd, check=True, capture_output=True)

    c_lib = ctypes.CDLL(str(lib_path))
    c_fn = c_lib.compute_pair_forces
    ptr = ctypes.POINTER(ctypes.c_double)
    c_fn.argtypes = [ctypes.c_int, ptr, ptr, ctypes.c_double, ctypes.c_double, ctypes.c_double]
    c_fn.restype = None

    def pair_forces_c(pos, k=4000.0, r0=0.70, rc=0.70):
        pos_c = np.ascontiguousarray(pos, dtype=np.float64)
        F = np.empty_like(pos_c)
        c_fn(
            ctypes.c_int(pos_c.shape[0]),
            pos_c.ctypes.data_as(ptr),
            F.ctypes.data_as(ptr),
            ctypes.c_double(rc),
            ctypes.c_double(r0),
            ctypes.c_double(k),
        )
        return F

    _C_PAIR_FORCES_FN = pair_forces_c
    return pair_forces_c
