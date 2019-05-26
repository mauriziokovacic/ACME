import math

def ratio(N):
    """
    Returns the N-th ratio value

    Parameters
    ----------
    N : int
        the ratio type

    Returns
    -------
    float
        the ratio value
    """

    return (N+math.sqrt(N*N+4))/2



def golden_ratio():
    """
    Returns the golden ratio value

    Returns
    -------
    float
        the ratio value
    """

    return ratio(1)



def silver_ratio():
    """
    Returns the silver ratio value

    Returns
    -------
    float
        the ratio value
    """

    return ratio(2)



def bronze_ratio():
    """
    Returns the bronze ratio value

    Returns
    -------
    float
        the ratio value
    """

    return ratio(3)
