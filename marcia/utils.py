def load_data_once(func):
    data = None

    def wrapper(*args, **kwargs):
        nonlocal data
        if data is None:
            data = func(*args, **kwargs)
        return data

    return wrapper