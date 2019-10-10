def homo2cart(P):
    """
    Converts a points set from homogeneous coordinates to standard cartesian

    Parameters
    ----------
    P : Tensor
        the affine coordinates

    Returns
    -------
    Tensor
        the cartesian coorodinates
    """

    return P[:, :-1]
