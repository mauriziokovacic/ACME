import torch
from ..math.cross      import *
from ..math.normvec    import *
from ..topology.ispoly import *
from .area             import *


def gradient(P, T, F):
    """
    Computes the gradient of a scalar field F over the input triangle mesh

    Parameters
    ----------
    P : Tensor
        the input points set tensor
    T : LongTensor
        the topology tensor
    F : Tensor
        the scalar field

    Returns
    -------
    Tensor
        the gradient of the input scalar field

    Raises
    ------
    AssertionError
        if the topology is not triangle
    """

    assert istri(T), 'Unsupported topology'
    Pi, Pj, Pk = P[T]
    Fi, Fj, Fk = F[T]
    Eij = Pj-Pi
    Ejk = Pk-Pj
    Eki = Pi-Pk
    N   = normr(cross(Eij, -Eki))
    A   = triangle_area(P, T)
    dF  = (Fi * cross(N, Ejk) +
           Fj * cross(N, Eki) +
           Fk * cross(N, Eij)) * torch.reciprocal(torch.mul(A, 2))
    return -dF
