def bary2data(P, T, B, t):
    """
    Returns the interpolated data using the given barycentric coordinates over the input geometry

    Parameters
    ----------
    P : Tensor
        the (N,D,) points set tensor
    T : LongTensor
        the (F,M) topology tensor
    B : Tensor
        the (X,F,) barycentric coordinates tensor
    t : LongTensor
        the (X,) faces indices tensor

    Returns
    -------
    Tensor
        the (X,D,) interpolated points set tensor
    """

    return P[T[:, t]] * B.unsqueeze(1)
