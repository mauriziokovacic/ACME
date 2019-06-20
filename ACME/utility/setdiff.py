import torch
from .flatten import *

def setdiff(A,B):
    """
    Returns the set difference tensor with all the elements in A not belonging to B

    Parameters
    ----------
    A : Tensor
        the first input tensor
    B : Tensor
        the second input tensor

    Returns
    -------
    Tensor
        the set difference tensor
    """

    return torch.tensor(list(set(flatten(A))-set(flatten(B))),dtype=A.dtype,device=A.device)
