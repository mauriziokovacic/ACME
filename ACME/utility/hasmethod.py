def hasmethod(obj, name):
    if hasattr(obj, name):
        return callable(getattr(obj, name))
    return False
