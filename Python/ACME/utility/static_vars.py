def static_vars(**kwargs):
    """
    Decorates a function with the given attributes

    Parameters
    ----------
    **kwargs
        a list of attributes and their initial value
    """

    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate
