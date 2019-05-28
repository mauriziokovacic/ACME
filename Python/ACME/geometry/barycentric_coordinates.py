import torch
from utility.row import *
from utility.col import *
from math.dot    import *



def barycentric_coordinates(Pi,Pj,Pk,Q):
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
        the point to compute the barycentric coordinates

    Returns
    -------
    Tensor
        the barycentric coordinates
    """

    B      = torch.zeros(row(Q),col(Q),
                         dtype=torch.float,
                         device=Q.device)
    Ei     = Pk - Pi
    Ej     = Pj - Pi
    Ek     = Q  - Pi
    Dii    = dot(Ei,Ei)
    Dij    = dot(Ei,Ej)
    Dik    = dot(Ei,Ek)
    Djj    = dot(Ej,Ej)
    Djk    = dot(Ej,Ek)
    d      = torch.reciprocal(Dii*Djj-Dij*Dij)
    B[:,2] = (Djj*Dik-Dij*Djk)*d
    B[:,1] = (Dii*Djk-Dij*Dik)*d
    B[:,0] = 1-B[:,1]-B[:,2]
    return B
