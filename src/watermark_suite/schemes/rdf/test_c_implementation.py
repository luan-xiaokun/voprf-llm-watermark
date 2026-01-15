import numpy as np
import time
import sys
import os

# Ensure we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

# Import C wrapper
from src.watermark_suite.schemes.rdf.levenshtein_c_wrapper import detect_c

# Import Cython version logic if possible, or just reimplement pure python logic for small test
# We can import the actual cython module if it compiles, but user said it fails.
# Let's try to import the cython one, likely it works now if we fixed the error, or at least the pure python one.
# For verification, I'll write a simple pure python implementation of the logic to compare results on small data.


def detect_python(tokens, n, k, xi, gamma=0.0):
    m = len(tokens)
    if k > m:
        return float("inf")

    vocab_size = xi.shape[1]
    import math

    min_val = float("inf")

    # Simple Python implementation
    # Note: xi logic: xi[(start_row + j - 1) % n_total_rows, tokens[token_start_idx + i - 1]]

    for i in range(m - k + 1):
        for j in range(n):
            # Calculate dist for this window & shift

            # Init buffer
            # (k+1) x (k+1)
            # Just use 2 rows optimization not needed for reference code
            dp = np.zeros((k + 1, k + 1), dtype=np.float32)

            for ii in range(k + 1):
                dp[ii, 0] = ii * gamma
            for jj in range(k + 1):
                dp[0, jj] = jj * gamma

            for ii in range(1, k + 1):
                for jj in range(1, k + 1):
                    xi_row = (j + jj - 1) % n
                    token_val = tokens[i + ii - 1]
                    xi_val = xi[xi_row, token_val]
                    cost = math.log(1 - xi_val)

                    val_ins = dp[ii - 1, jj] + gamma
                    val_del = dp[ii, jj - 1] + gamma
                    val_sub = dp[ii - 1, jj - 1] + cost

                    dp[ii, jj] = min(val_ins, val_del, val_sub)

            res = dp[k, k]
            if res < min_val:
                min_val = res

    return min_val


def test_correctness():
    print("Testing Correctness C vs Python...")
    np.random.seed(42)

    n = 256
    k = 50
    m = 60  # small diff
    vocab_size = 1000

    tokens = np.random.randint(0, vocab_size, size=m, dtype=np.int64)
    # Ensure xi values are small enough so 1-xi > 0
    xi = np.random.rand(n, vocab_size).astype(np.float32) * 0.5

    # Run Python
    start = time.time()
    res_py = detect_python(tokens, n, k, xi)
    print(f"Python result: {res_py:.6f}, time: {time.time() - start:.4f}s")

    # Run C
    start = time.time()
    res_c = detect_c(tokens, n, k, xi)
    print(f"C result:      {res_c:.6f}, time: {time.time() - start:.4f}s")

    if abs(res_py - res_c) < 1e-4:
        print("✅ Correctness PASSED")
    else:
        print("❌ Correctness FAILED")
        exit(1)


def test_performance():
    print("\nTesting Performance C...")
    np.random.seed(42)
    n = 256
    k = 256
    m = 256  # typical window
    vocab_size = 50000

    tokens = np.random.randint(0, vocab_size, size=m, dtype=np.int64)
    xi = np.random.rand(n, vocab_size).astype(np.float32) * 0.5

    # Warmup
    detect_c(tokens, n, k, xi)

    start = time.time()
    loops = 10
    for _ in range(loops):
        detect_c(tokens, n, k, xi)
    end = time.time()

    avg_time = (end - start) / loops
    print(f"Reference dimensions (m={m}, n={n}, k={k}): {avg_time:.4f} s/run")


if __name__ == "__main__":
    test_correctness()
    test_performance()
