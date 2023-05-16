import numpy as np
from numba import jit
from functools import wraps

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


def temporarily_false(attribute_name):
    def decorator(func):
        @wraps(func)
        def wrapper(instance, *args, **kwargs):
            original_value = getattr(instance, attribute_name)
            if original_value:
                setattr(instance, attribute_name, False)
                result = func(instance, *args, **kwargs)
                setattr(instance, attribute_name, original_value)
            else:
                result = func(instance, *args, **kwargs)
            return result
        return wrapper
    return decorator
