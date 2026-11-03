#!/usr/bin/env python3
"""Print GPU 0's name, SM count and compute capability, read from the CUDA driver."""
import ctypes

cu = ctypes.CDLL("libcuda.so.1")
cu.cuInit(0)
dev = ctypes.c_int()
cu.cuDeviceGet(ctypes.byref(dev), 0)
name = ctypes.create_string_buffer(100)
cu.cuDeviceGetName(name, 100, dev)


def attr(k):
    v = ctypes.c_int()
    cu.cuDeviceGetAttribute(ctypes.byref(v), k, dev)
    return v.value


# 16 multiprocessor count, 75/76 compute capability major/minor, 8 shared memory per block
print(f"{name.value.decode()} sm={attr(16)} cc={attr(75)}.{attr(76)} smem_per_block={attr(8)}")
