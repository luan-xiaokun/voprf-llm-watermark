import cython

# 使用 cdef class 将其定义为 C 扩展类型
cdef class MersenneRNG:
    # --- 1. 将所有属性声明为 C 级别变量 ---
    # 使用 C 的无符号整型数组，速度极快
    cdef unsigned int state[624]
    cdef unsigned int f, m, u, s, b, t, c, l
    cdef int index
    cdef unsigned int lower_mask, upper_mask

    # __init__ 仍然是 Python 可调用的方法
    def __init__(self, unsigned int seed=5489):
        # --- 2. 声明 C 级别的循环变量 ---
        cdef int i

        # --- 3. 初始化 C 级别的属性 ---
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
            # --- 4. 直接在 C 层面进行计算，无需 int_32 转换 ---
            # C 的 unsigned int 会自动处理 32 位溢出
            self.state[i] = self.f * (self.state[i - 1] ^ (self.state[i - 1] >> 30)) + i

    # cdef 方法是纯 C 函数，速度最快，只能被其他 cdef/cpdef 方法调用
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

    # cpdef 方法同时提供 C 接口和 Python 接口，非常灵活
    cpdef unsigned int randint(self):
        cdef unsigned int y

        if self.index >= 624:
            self.twist()  # 调用快速的 cdef 方法
        
        y = self.state[self.index]
        y = y ^ (y >> self.u)
        y = y ^ ((y << self.s) & self.b)
        y = y ^ ((y << self.t) & self.c)
        y = y ^ (y >> self.l)
        
        self.index += 1
        return y

    cpdef double rand(self):
        # C 级别的 double 类型，避免 Python float 对象开销
        return self.randint() * (1.0 / 4294967296.0)

    # 这个方法返回 Python 列表，所以它必须是 def 方法
    # 但内部的 randint 调用会非常快
    def randperm(self, int n):
        cdef int i, j
        cdef list p = list(range(n))
        for i in range(n - 1, 0, -1):
            # 调用快速的 cpdef 方法
            j = self.randint() % i
            p[i], p[j] = p[j], p[i]
        return p