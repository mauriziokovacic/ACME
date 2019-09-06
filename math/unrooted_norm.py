import torch
from ..utility.isinf import *
from ..utility.strcmpi import *


def unrooted_norm(tensor, p=2, dim=1):
    """
    Computes the unrooted norm of the input tensor along the given dimension.

    Parameters
    ----------
    tensor : Tensor
        the input tensor
    p : 1,2 or 'inf' (optional)
        the exponent of the norm. Only the three specified values are supported (default is 2)
    dim : int (optional)
        dimension along the norm is computed (default is 1)

    Returns
    -------
    Tensor
        a tensor containing the unrooted norm of the input tensor

    Raises
    ------
    ValueError
        if p is different than the supported values
    """

    if p == 2:
        return torch.sum(torch.pow(tensor, p), dim, keepdim=True)
    if p == 1:
        return torch.sum(torch.abs(tensor), dim, keepdim=True)
    if isinf(p) or strcmpi(p, 'inf'):
        return torch.max(torch.abs(tensor), dim, keepdim=True)[0]
    raise ValueError('Only 1-norm, 2-norm and Inf-norm are supported.')


def sqnorm(tensor, dim=1):
    """
    Computes the squared norm of the input tensor along the given dimension.

    Parameters
    ----------
    tensor : Tensor
        the input tensor
    dim : int (optional)
        dimension along the norm is computed (default is 1)

    Returns
    -------
    Tensor
        a tensor containing the squared norm of the input tensor
    """

    return unrooted_norm(tensor, p=2, dim=dim)


def sqdistance(A, B, dim=1):
    """
    Computes the squared distance between the given tensors

    Parameters
    ----------
    A : Tensor
        first tensor
    B : Tensor
        second tensor
    dim : int (optional)
        dimension along the norm is computed (default is 1)

    Returns
    -------
    Tensor
        a tensor containing the norm of the input tensors
    """

    return sqnorm(A-B, dim=dim)
