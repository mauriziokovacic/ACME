import torch
from ..utility.row     import *
from ..utility.col     import *
from ..utility.indices import *
from .normw            import *


def limit_weight(W, k=4):
    """
    Limits the number of non-zero weights to k

    Parameters
    ----------
    W : Tensor
        the (N,H,) weights matrix
    k : int (optional)
        the target number of non zero weights (default is 4)

    Returns
    -------
    Tensor
        the (N,H,) limited weights matrix
    """

    w, J = torch.sort(W, dim=1, descending=True)
    w    = normw(w[:, :k])
    J    = J[:, :k]
    I    = indices(0, row(W)-1, device=W.device)
    out  = torch.zeros(row(W), col(W), dtype=W.dtype, device=W.device)
    for i, j, wk in zip(I, J, w):
        out[I, J] = wk
    return out
