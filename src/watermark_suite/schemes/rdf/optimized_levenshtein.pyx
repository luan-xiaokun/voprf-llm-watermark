# filename: optimized_levenshtein.pyx

import numpy as np
cimport numpy as np
cimport cython

from cython.parallel cimport prange
from libc.math cimport log, INFINITY
from libc.stdlib cimport malloc, free
from libc.string cimport memset

cdef extern from "omp.h" nogil:
    int omp_get_max_threads()
    int omp_get_thread_num()


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
) nogil:
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
    cdef int i, j, thread_id, n_threads, buffer_size
    cdef float val, local_min_val
    cdef float* thread_min_vals_ptr = NULL
    cdef float* A_buffer_ptr = NULL
    cdef float final_min_val = INFINITY

    # ✅ buffer_size 是常量，在循环外计算一次即可
    buffer_size = (k + 1) * (k + 1)

    try:
        n_threads = omp_get_max_threads()
        thread_min_vals_ptr = <float*> malloc(n_threads * sizeof(float))
        if not thread_min_vals_ptr:
            raise MemoryError()

        for i in range(n_threads):
            thread_min_vals_ptr[i] = INFINITY

        # ✅ prange 循环体内不再有任何 cdef 语句
        for i in prange(m - (k - 1), nogil=True, schedule='static'):
            A_buffer_ptr = <float*> malloc(buffer_size * sizeof(float))
            if not A_buffer_ptr:
                continue

            memset(A_buffer_ptr, 0, buffer_size * sizeof(float))
            
            local_min_val = INFINITY
            thread_id = omp_get_thread_num()

            for j in range(n):
                val = _levenshtein_cython_c(
                    tokens, i, k, xi, j, n, gamma, A_buffer_ptr
                )
                if val < local_min_val:
                    local_min_val = val
            
            if local_min_val < thread_min_vals_ptr[thread_id]:
                thread_min_vals_ptr[thread_id] = local_min_val

            free(A_buffer_ptr)
            A_buffer_ptr = NULL

        for i in range(n_threads):
            if thread_min_vals_ptr[i] < final_min_val:
                final_min_val = thread_min_vals_ptr[i]
        
        return final_min_val

    finally:
        if thread_min_vals_ptr != NULL:
            free(thread_min_vals_ptr)