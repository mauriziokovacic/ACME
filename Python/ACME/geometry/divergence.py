import torch
from utility.row         import *
from utility.accumarray  import *
from math.dot            import *
from topology.poly2ind   import *
from .triangle_cotangent import *

def divergence(P,T,dF):
    """
    Computes the divergence of a vector field dF over the input triangle mesh

    Parameters
    ----------
    P : Tensor
        the input points set tensor
    T : LongTensor
        the topology tensor
    dF : Tensor
        the vector field

    Returns
    -------
    Tensor
        the divergence of the input vector field

    Raises
    ------
    AssertionError
        if the topology is not triangle
    """


    assert istri(T), 'Unsupported topology'
    n        = row(P)
    I,J,K    = tri2ind(T)
    Pi,Pj,Pk = P[T]
    Ci,Cj,Ck = triangle_cotangent(P,T)
    Eij      = Pj-Pi
    Ejk      = Pk-Pj
    Eki      = Pi-Pk
    Di       = accumarray(I, CTk * dot(Eij,dF) + CTj * dot(-Eki,dF), n)
    Dj       = accumarray(J, CTi * dot(Ejk,dF) + CTk * dot(-Eij,dF), n)
    Dk       = accumarray(K, CTj * dot(Eki,dF) + CTi * dot(-Ejk,dF), n)
    return torch.mul(Di+Dj+Dk, 0.5)
