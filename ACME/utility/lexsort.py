import torch
from .indices import *


def lexsort(A, dim=1):
    """
    Lexicographic sort of elements in a given Tensor

    Parameters
    ----------
    A : Tensor
        a two dimensional Tensor
    dim : int (optional)
        the sorting dimension (default is 1)

    Returns
    -------
    (Tensor,LongTensor)
        The sorted Tensor and the rows/cols indices in the original input
    """

    out = A.clone()
    if dim == 0:
        out = torch.t(out)
    off = indices(out.shape[1]-1, 0, dtype=A.dtype, device=A.device).squeeze()*out.shape[0]+1
    i   = torch.argsort(torch.sum(out*off, dim=1))
    if dim == 0:
        out = A[:, i]
    else:
        out = A[i]
    return out, i
