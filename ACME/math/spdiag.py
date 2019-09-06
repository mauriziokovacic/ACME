import torch
from ..utility.numel        import *
from ..utility.indices      import *
from ..utility.LongTensor   import *
from ..utility.SparseTensor import *


def spdiag(tensor, rows=None, cols=None):
    """
    Creates a sparse diagonal matrix from the given input tensor

    Parameters
    ----------
    tensor : Tensor
        the input tensor
    rows : int (optional)
        the number of rows in the matrix. If None it will be automatically computed (default is None)
    cols : int (optional)
        the number of cols in the matrix. If None it will be automatically computed (default is None)

    Returns
    -------
    SparseTensor
        a sparse diagonal matrix with the given entries on the diagonal
    """

    if rows is None:
        rows = numel(tensor)
    if cols is None:
        cols = rows
    n = min(rows, cols)
    v = torch.cat((tensor.flatten(), torch.zeros(n-numel(tensor), dtype=tensor.dtype, device=tensor.device)))
    E = torch.t(indices(0, n, device=tensor.device))
    return SparseTensor(size=(rows, cols), indices=torch.cat((E,E),dim=0), values=v)
