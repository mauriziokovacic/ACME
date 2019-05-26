import torch
from math.cross import *
from math.norm  import *

def triangle_area(P,T):
    """
    Returns the triangle area of all the input triangles

    Parameters
    ----------
    P : Tensor
        a (n,3) point set tensor
    T : LongTensor
        the topology tensor

    Returns
    -------
    Tensor
        the triangle areas
    """

    Pi,Pj,Pk = P[T]
    return torch.mul(norm(cross(Pj-Pi,Pk-Pi,dim=1),dim=1),0.5)
