import torch
from .norm import *


def normvec(tensor, p=2, dim=1):
    """
    Normalizes the input tensor along the specified dimension, using the specified norm

    Parameters
    ----------
    tensor : Tensor
        the input tensor
    p : 1,2 or 'inf' (optional)
        exponent of the norm (default is 2)
    dim : int (optional)
        dimension along the norm is computed (default is 1)
    """

    n = pnorm(tensor, p=p, dim=dim)
    n[n == 0] = 1
    return tensor / n
    return tensor/(n+torch.eq(n, 0).to(dtype=torch.float, device=tensor.device))



def normr(tensor, p=2):
    """
    Normalizes the input tensor rows using the specified norm

    Parameters
    ----------
    tensor : Tensor
        the input tensor
    p : 1,2 or 'inf' (optional)
        exponent of the norm (default is 2)
    """

    return normvec(tensor, p=p, dim=-1)


def normc(tensor, p=2):
    """
    Normalizes the input tensor cols using the specified norm

    Parameters
    ----------
    tensor : Tensor
        the input tensor
    p : 1,2 or 'inf' (optional)
        exponent of the norm (default is 2)
    """

    return normvec(tensor, p=p, dim=-2)


def normd(tensor, p=2):
    """
    Normalizes the input tensor channels/slices using the specified norm

    Parameters
    ----------
    tensor : Tensor
        the input tensor
    p : 1,2 or 'inf' (optional)
        exponent of the norm (default is 2)
    """

    return normvec(tensor, p=p, dim=2)
