import numpy as np
from numba import jit

def load_data_once(func):
    data = None
    def wrapper(*args, **kwargs):
        nonlocal data
        if data is None:
            data = func(*args, **kwargs)
        return data
    return wrapper

@jit(nopython=True)
def nan_to_zero(arr):
    result = arr.copy()
    for i in range(result.shape[0]):
        if np.isnan(result[i]):
            result[i] = 0
    return result
