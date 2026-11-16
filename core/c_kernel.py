"""Native C pairforce kernel loader."""

import ctypes
import platform
import shutil
import subprocess
from pathlib import Path
import numpy as np

_C_PAIR_FORCES_FN = None


# Compile and load pairforce.c shared library via ctypes
def get_c_pair_forces_fn():
    global _C_PAIR_FORCES_FN
    if _C_PAIR_FORCES_FN is not None:
        return _C_PAIR_FORCES_FN

    core_dir = Path(__file__).resolve().parent
    root_dir = core_dir.parent

    # Candidate source paths
    candidates = [
        core_dir / "pairforce.c",
        root_dir / "pairforce.c",
        root_dir / "Task6_Computational_Efficiency" / "pairforce_task6.c",
    ]
    c_source = None
    for cand in candidates:
        if cand.exists():
            c_source = cand
            break

    if c_source is None:
        raise FileNotFoundError("Could not find pairforce.c.")

    # Platform shared library extension
    system = platform.system()
    lib_name = (
        "pairforce.dll"
        if system == "Windows"
        else ("libpairforce.dylib" if system == "Darwin" else "libpairforce.so")
    )
    lib_path = core_dir / lib_name

    # Compile if library binary does not exist
    if not lib_path.exists():
        compiler = "clang" if shutil.which("clang") and system == "Darwin" else "gcc"
        if not shutil.which(compiler):
            compiler = shutil.which("gcc") or shutil.which("clang")
        if not compiler:
            raise RuntimeError("No C compiler (gcc or clang) found on PATH.")

        flags = ["-O2", "-shared"] if system != "Darwin" else ["-O2", "-dynamiclib", "-fPIC"]
        cmd = [compiler] + flags + ["-o", str(lib_path), str(c_source), "-lm"]
        subprocess.run(cmd, check=True, capture_output=True)

    # Bind C function
    c_lib = ctypes.CDLL(str(lib_path))
    c_fn = c_lib.compute_pair_forces
    ptr = ctypes.POINTER(ctypes.c_double)
    c_fn.argtypes = [ctypes.c_int, ptr, ptr, ctypes.c_double, ctypes.c_double, ctypes.c_double]
    c_fn.restype = None

    # Python wrapper function
    def pair_forces_c(pos, k=4000.0, r0=0.70, rc=0.70):
        N = len(pos)
        pos_c = np.ascontiguousarray(pos, dtype=np.float64)
        F_c = np.zeros((N, 2), dtype=np.float64)

        pos_ptr = pos_c.ctypes.data_as(ptr)
        F_ptr = F_c.ctypes.data_as(ptr)

        c_fn(N, pos_ptr, F_ptr, float(rc), float(r0), float(k))

        if pos.dtype == np.float32:
            return F_c.astype(np.float32)
        return F_c

    _C_PAIR_FORCES_FN = pair_forces_c
    return _C_PAIR_FORCES_FN
