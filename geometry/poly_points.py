def poly_points(P, T):
    """
    Returns the ordered points from the given polygons

    Parameters
    ----------
    P : Tensor
        a (N, D,) points set tensor
    T : LongTensor
        a (M, T,) topology tensor

    Returns
    -------
    tuple
        a tuple containing the points of the given polygons
    """

    return tuple(P[T])
