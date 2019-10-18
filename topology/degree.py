import torch
from ..utility.ismatrix import *
from ..utility.issparse import *
from ..math.spdiag      import *


def degree(A):
    """
    Returns the degree matrix from the given adjacency matrix

    Parameters
    ----------
    A : Tensor
        the input adjacency matrix

    Returns
    -------
    Tensor
        the degree matrix

    Raises
    ------
    AssertionError
        if input is not a matrix
    """

    assert ismatrix(A), 'Tensor must be a matrix'
    if issparse(A):
        return spdiag(torch.sparse.sum(A, dim=-1).squeeze().to_dense())
    return torch.diag(torch.sum(A, -1, keepdim=False))
