import torch
from ..utility.isinf import *
from .knn            import *
from .unrooted_norm  import *


def pnorm(V, p=2, dim=1):
    """
    Computes the norm of the input tensor along the given dimension.

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
        a tensor containing the norm of the input tensor
    """

    n = unrooted_norm(V, p=p, dim=dim)
    if isinf(p) or (p == 'inf') or (p == 1):
        return n
    return torch.sqrt(n)


def norm(tensor, dim=1):
    """
    Computes the euclidean norm of the input tensor along the given dimension.

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

    return pnorm(tensor, p=2, dim=dim)


def hypot(x, y, dim=1):
    """
    Computes the euclidean norm of the input tensor [x,y] along the given dimension.

    The function concatenates the input tensors along the specified dimension and
    then proceeds to compute the euclidean norm along that dimension

    Parameters
    ----------
    x : Tensor
        x component of the tensor
    y : Tensor
        y component of the tensor
    dim : int (optional)
        dimension along the norm is computed (default is 1)

    Returns
    -------
    Tensor
        a tensor containing the squared norm of the input tensors
    """

    return pnorm(torch.cat((x, y), dim=dim), p=2, dim=dim)


def distance(A, B, p=2, dim=1):
    """
    Computes the distance between the given tensors, using the specified norm along the specified dimension

    Parameters
    ----------
    A : Tensor
        first tensor
    B : Tensor
        second tensor
    p : 1,2 or 'inf' (optional)
        the exponent of the norm. Only the three specified values are supported (default is 2)
    dim : int (optional)
        dimension along the norm is computed (default is 1)

    Returns
    -------
    Tensor
        a tensor containing the norm of the input tensors
    """

    return pnorm(A-B, p=p, dim=dim)


def hausdorff(A, B):
    """
    Computes the Hausdorff distance between the given points sets

    Parameters
    ----------
    A : Tensor
        a (M,D,) tensor
    B : Tensor
        a (N,D,) tensor

    Returns
    -------
    Tensor
        the (1,) tensor representing the Hausdorff distance
    """

    return max(knn(A, B, 1, distFcn=distance)[1].max(),
               knn(B, A, 1, distFcn=distance)[1].max())
