import torch
from math.norm import *

def normalize_scale(P):
    """
    Returns the rescaled points set in range [-1,1]

    Parameters
    ----------
    P : Tensor
        the input points set tensor

    Returns
    -------
    Tensor
        the rescaled points set
    """

    min = torch.min(P,dim=0,keepdim=True)[0]
    max = torch.max(P,dim=0,keepdim=True)[0]
    d   = torch.mul(distance(min,max),0.5)
    return torch.div(P,d)
