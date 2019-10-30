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

    n = numel(tensor)
    if rows is None:
        rows = n
    if cols is None:
        cols = rows
    return SparseTensor(size=(rows, cols),
                        indices=indices(0, n-1, device=tensor.device).expand(-1, 2),
                        values=tensor.view(-1))
