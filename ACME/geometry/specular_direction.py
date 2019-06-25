import torch
from ..math.dot import *

def specular_direction(dN,dI,dim=1):
    """
    Computes the specular vector of an input one w.r.t. a given direction

    Parameters
    ----------
    dN : Tensor
        the specular relative direction
    dI : Tensor
        the vector to compute the specular from
    dim : int (optional)
        the dimension along the direction will be computed (default is 1)

    Returns
    -------
    Tensor
        the specular direction
    """

    return torch.mul(dot(dN,dI,dim=dim),2)*dN-dI
