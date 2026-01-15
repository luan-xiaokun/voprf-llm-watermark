# filename: optimized_levenshtein.pyx
# Cython module for optimized Levenshtein distance detection


import numpy as np
cimport numpy as np
cimport cython

from cython.parallel cimport prange, parallel
from libc.math cimport log, INFINITY
from libc.stdlib cimport malloc, free
from libc.string cimport memset

cdef extern from "omp.h" nogil:
    int omp_get_max_threads()
    int omp_get_thread_num()
    int omp_get_num_threads()


@cython.boundscheck(False)
@cython.wraparound(False)
cdef float _levenshtein_cython_c(
    long[:] tokens,
    int token_start_idx,
    int k,
    float[:,:] xi,
    int start_row,
    int n_total_rows,
    float gamma,
    float* A_buffer_ptr
) noexcept nogil:
    cdef int i, j
    cdef float cost
    cdef int row_width = k + 1

    for i in range(k + 1):
        for j in range(k + 1):
            if i == 0:
                A_buffer_ptr[i * row_width + j] = j * gamma
            elif j == 0:
                A_buffer_ptr[i * row_width + j] = i * gamma
            else:
                cost = log(1 - xi[(start_row + j - 1) % n_total_rows, tokens[token_start_idx + i - 1]])
                A_buffer_ptr[i * row_width + j] = min(
                    A_buffer_ptr[(i - 1) * row_width + j] + gamma,
                    A_buffer_ptr[i * row_width + j - 1] + gamma,
                    A_buffer_ptr[(i - 1) * row_width + (j - 1)] + cost
                )

    return A_buffer_ptr[k * row_width + k]


@cython.boundscheck(False)
@cython.wraparound(False)
def detect_cython(
    long[:] tokens,
    int n,
    int k,
    float[:,:] xi,
    float gamma=0.0
):
    cdef int m = tokens.shape[0]
    if k > m:
        return INFINITY

    # ✅ 所有 C 变量都在 prange 循环外部声明
    cdef int i, j, thread_id, n_threads, buffer_size, idx, total_iters
    cdef int num_threads_actual, start_idx, end_idx
    cdef float val, local_min_val
    cdef float* thread_min_vals_ptr = NULL
    cdef float* A_buffer_ptr = NULL
    cdef float final_min_val = INFINITY

    # ✅ buffer_size 是常量，在循环外计算一次即可
    buffer_size = (k + 1) * (k + 1)
    
    total_iters = (m - k + 1) * n

    try:
        n_threads = omp_get_max_threads()
        thread_min_vals_ptr = <float*> malloc(n_threads * sizeof(float))
        # Allocating one large buffer for all threads to avoid malloc contention inside parallel region
        A_buffer_ptr = <float*> malloc(n_threads * buffer_size * sizeof(float))
        
        if not thread_min_vals_ptr or not A_buffer_ptr:
            raise MemoryError()

        for i in range(n_threads):
            thread_min_vals_ptr[i] = INFINITY

        with nogil, parallel():
            thread_id = omp_get_thread_num()
            num_threads_actual = omp_get_num_threads()
            
            # Each thread uses its own slice of the pre-allocated buffer
            # We use a local pointer to avoid pointer arithmetic in the inner loop (though optimizing compiler handles this)
            # Declaring a local pointer variable matching the type of A_buffer_ptr
            # We can't declare new variables here that are not in the declarations, but A_buffer_ptr is declared
            # Let's use a new local pointer variable logic
            # Re-using A_buffer_ptr variable inside thread? No, it's shared.
            # We need a private variable for the thread's buffer.
            # In Cython parallel, variables assigned to in the block are private by default if not shared?
            # Actually, we need to be careful.
            # Best pattern:
            # cdef float* my_buffer = A_buffer_ptr + thread_id * buffer_size
            pass

            local_min_val = INFINITY
            
            # Manual loop distribution
            start_idx = (thread_id * total_iters) // num_threads_actual
            end_idx = ((thread_id + 1) * total_iters) // num_threads_actual
            
            for idx in range(start_idx, end_idx):
                i = idx // n
                j = idx % n
                
                val = _levenshtein_cython_c(
                    tokens, i, k, xi, j, n, gamma, A_buffer_ptr + thread_id * buffer_size
                )
                
                if val < local_min_val:
                    local_min_val = val
            
            # Write to shared memory only once per thread
            if local_min_val < thread_min_vals_ptr[thread_id]:
                thread_min_vals_ptr[thread_id] = local_min_val
                
        for i in range(n_threads):
            if thread_min_vals_ptr[i] < final_min_val:
                final_min_val = thread_min_vals_ptr[i]
        
        return final_min_val

    finally:
        if thread_min_vals_ptr != NULL:
            free(thread_min_vals_ptr)
        if A_buffer_ptr != NULL:
            free(A_buffer_ptr)