from ..utility.sum import *


def point_plane_distance(P, N, Q, dim=-1):
    """
    Returns the distance of point Q from the plane passing in P, with normal N

    Parameters
    ----------
    P : Tensor
        a point on the plane
    N : Tensor
        the normal of the plane
    Q : Tensor
        a point to compute the distance
    dim : int (optional)
        the dimension along the distance is computed

    Returns
    -------
    Tensor
        the point-plane distances
    """

    return sum(N*(P-Q), dim, keepdim=True)
