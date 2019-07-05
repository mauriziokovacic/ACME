def affine2cart(P):
    """
    Converts a points set from affine coordinates to standard cartesian

    Parameters
    ----------
    P : Tensor
        the affine coordinates

    Returns
    -------
    Tensor
        the cartesian coorodinates
    """

    return P[:, 0:3]
