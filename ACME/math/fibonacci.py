import math


def n_bonacci(N, n):
    """
    Computes the n-bonacci number for the given input

    Parameters
    ----------
    N : int
        the sequence number
    n : int
        the number to compute the series from

    Returns
    -------
    int
        the n-bonacci number for the given input
    """

    if n <= 1:
        return n
    return N*n_bonacci(n-1, N)+n_bonacci(n-2, N)


def fibonacci(n):
    """
    Computes the fibonacci number for the given input

    Parameters
    ----------
    n : int
        the number to compute the series from

    Returns
    -------
    int
        the fibonacci number
    """

    return n_bonacci(n, 1)
