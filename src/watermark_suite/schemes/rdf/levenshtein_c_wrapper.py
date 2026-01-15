import ctypes
import numpy as np
import os
import sys

# Load the shared library
try:
    _lib_path = os.path.join(os.path.dirname(__file__), "optimized_levenshtein_c.so")
    _lib = ctypes.CDLL(_lib_path)

    # Define signature
    # float detect_c(const int64_t* tokens, int m, int n, int k, const float* xi, int vocab_size, float gamma)
    _lib.detect_c.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags="C_CONTIGUOUS"),  # tokens
        ctypes.c_int,  # m
        ctypes.c_int,  # n
        ctypes.c_int,  # k
        np.ctypeslib.ndpointer(
            dtype=np.float32, ndim=2, flags="C_CONTIGUOUS"
        ),  # xi (passed as ptr, ndim 2 ok if c_contiguous)
        ctypes.c_int,  # vocab_size
        ctypes.c_float,  # gamma
    ]
    _lib.detect_c.restype = ctypes.c_float

    def detect_c(tokens, n, k, xi, gamma=0.0):
        # Ensure correct types
        if not isinstance(tokens, np.ndarray) or tokens.dtype != np.int64:
            tokens = np.array(tokens, dtype=np.int64)
        if not tokens.flags["C_CONTIGUOUS"]:
            tokens = np.ascontiguousarray(tokens)

        if not isinstance(xi, np.ndarray) or xi.dtype != np.float32:
            xi = np.array(xi, dtype=np.float32)
        if not xi.flags["C_CONTIGUOUS"]:
            xi = np.ascontiguousarray(xi)

        m = len(tokens)
        vocab_size = xi.shape[1]
        if xi.shape[0] != n:
            # Just a warning or check, though the C code relies on passed n for iteration logic but xi shape for access
            pass

        return _lib.detect_c(tokens, m, n, k, xi, vocab_size, gamma)

except OSError as e:
    print(f"Warning: Could not load optimized_levenshtein_c.so: {e}", file=sys.stderr)
    detect_c = None
except Exception as e:
    print(f"Warning: Error loading C library: {e}", file=sys.stderr)
    detect_c = None
