import numpy as np
import time
import sys

def measure_size_lst(length=1):
    lst = list(range(length))
    total_size = sys.getsizeof(lst) + (length * sys.getsizeof(0))
    return total_size

def measure_size_ray(length=1):
    arr = np.arange(length, dtype=np.int32)
    return arr.nbytes + sys.getsizeof(arr)

def sqrt_elements_lst(low, high, length=1):
    start_time = time.time()
    lst = np.random.randint(low, high, length).tolist()
    sqrt_lst = []
    for x in lst:
        sqrt_lst.append(x ** 0.5)
    return time.time() - start_time

def sqrt_elements_numpy(low, high, length=1):
    start_time = time.time()
    arr = np.random.randint(low, high, length)
    sqrt_arr = np.sqrt(arr)
    return time.time() - start_time