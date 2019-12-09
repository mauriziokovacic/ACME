from .point_plane_distance import *


def project_point_on_plane(P, N, Q, dim=1):
    """
    Returns the projection of a given point Q onto the plane passing in P, with normal N

    Parameters
    ----------
    P : Tensor
        a point on the plane
    N : Tensor
        the normal of the plane
    Q : Tensor
        the point to be projected
    dim : int (optional)
        the dimension along the projection is computed (default is 1)

    Returns
    -------
    Tensor
        the projected point
    """

    return Q-point_plane_distance(P, N, Q, dim=dim)*N

