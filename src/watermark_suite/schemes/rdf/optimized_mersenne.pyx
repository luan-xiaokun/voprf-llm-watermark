import cython
import numpy as np
cimport numpy as cnp

cdef class MersenneRNG:
    cdef unsigned int state[624]
    cdef unsigned int f, m, u, s, b, t, c, l
    cdef int index
    cdef unsigned int lower_mask, upper_mask

    def __init__(self, unsigned int seed=5489):
        cdef int i

        self.f = 1812433253
        self.m = 397
        self.u = 11
        self.s = 7
        self.b = 0x9D2C5680
        self.t = 15
        self.c = 0xEFC60000
        self.l = 18
        self.index = 624
        self.lower_mask = (1 << 31) - 1
        self.upper_mask = 1 << 31

        self.state[0] = seed
        for i in range(1, 624):
            self.state[i] = self.f * (self.state[i - 1] ^ (self.state[i - 1] >> 30)) + i

    cdef void twist(self):
        cdef int i
        cdef unsigned int temp, temp_shift

        for i in range(624):
            temp = (self.state[i] & self.upper_mask) + (self.state[(i + 1) % 624] & self.lower_mask)
            temp_shift = temp >> 1
            if temp % 2 != 0:
                temp_shift = temp_shift ^ 0x9908B0DF
            self.state[i] = self.state[(i + self.m) % 624] ^ temp_shift
        self.index = 0

    cpdef unsigned int randint(self):
        cdef unsigned int y

        if self.index >= 624:
            self.twist()
        
        y = self.state[self.index]
        y = y ^ (y >> self.u)
        y = y ^ ((y << self.s) & self.b)
        y = y ^ ((y << self.t) & self.c)
        y = y ^ (y >> self.l)
        
        self.index += 1
        return y

    cpdef double rand(self):
        return self.randint() * (1.0 / 4294967296.0)

    cpdef object random_array(self, Py_ssize_t count):
        cdef cnp.ndarray[cnp.float32_t, ndim=1] values
        cdef Py_ssize_t index
        if count < 0:
            raise ValueError("count must be non-negative")
        values = np.empty(count, dtype=np.float32)
        for index in range(count):
            values[index] = self.randint() * (1.0 / 4294967296.0)
        return values

    def randperm(self, int n):
        cdef int i, j
        cdef list p = list(range(n))
        for i in range(n - 1, 0, -1):
            j = self.randint() % i
            p[i], p[j] = p[j], p[i]
        return p
