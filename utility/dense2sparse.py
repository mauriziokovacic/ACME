from .issparse     import *
from .SparseTensor import *


def dense2sparse(A):
    """
    Given a tensor, it returns is sparse representation

    Parameters
    ----------
    A : Tensor
        a tensor

    Returns
    -------
    SparseTensor
        a sparse tensor
    """

    if issparse(A):
        return A
    s = list(A.shape)
    i = torch.nonzero(A)
    v = A[tuple(i.t())]
    return SparseTensor(size=s, indices=i, values=v)
