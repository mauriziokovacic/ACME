def barrier_function(x,t):
    """
    Returns the barrier function for a given value x and a threshold t

    Parameters
    ----------
    x : scalar or Tensor
        the input value
    t : scalar
        the threshold value from which the barrier starts

    Returns
    -------
    scalar or Tensor
        the barrier vlaue
    """

    return (x**3)/(t**3) - 3*(x**2)/(t**2) + 3*x/t
