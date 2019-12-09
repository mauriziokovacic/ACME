import torch
from ..utility.row import *
from ..utility.col import *
from ..math.dot    import *
from .BaryCoord    import *


def barycentric_coordinates(Pi, Pj, Pk, Q):
    """
    Returns the barycentric coordinates of point Q w.r.t. triangle Pi-Pj-Pk

    Parameters
    ----------
    Pi : Tensor
        the first point of the triangle
    Pj : Tensor
        the second point of the triangle
    Pk : Tensor
        the third point of the triangle
    Q : Tensor
        the point to compute the barycentric coordinates for

    Returns
    -------
    Tensor
        the barycentric coordinates
    """

    Ei      = Pk - Pi
    Ej      = Pj - Pi
    Ek      = Q  - Pi
    Dii     = dot(Ei, Ei, dim=-1)
    Dij     = dot(Ei, Ej, dim=-1)
    Dik     = dot(Ei, Ek, dim=-1)
    Djj     = dot(Ej, Ej, dim=-1)
    Djk     = dot(Ej, Ek, dim=-1)
    d       = torch.reciprocal(Dii * Djj - Dij * Dij)
    bk      = ((Djj * Dik - Dij * Djk) * d).squeeze()
    bj      = ((Dii * Djk - Dij * Dik) * d).squeeze()
    bi      = 1 - bj - bk
    if Q.ndimension() == 1:
        B = BaryCoord(Q.shape[0], device=Q.device)
        B[0] = bi
        B[1] = bj
        B[2] = bk
    else:
        B = BaryCoord(Q.shape[0], 3, device=Q.device)
        B[:, 0] = bi
        B[:, 1] = bj
        B[:, 2] = bk
    return B
