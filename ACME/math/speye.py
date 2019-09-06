from ..utility.row import *
from ..utility.col import *
from .spdiag       import *


def speye(rows, cols=None, dtype=torch.float, device='cuda:0'):
    """
    Creates a sparse identity matrix with the specified shape.
    If columns are not specified the matrix is intended squared.

    Parameters
    ----------
    rows : int
        the rows of the matrix
    cols : int (optional)
        the columns of the matrix
    dtype : type (optional)
        the type of the values (default is torch.float)
    device : str or torch.device (optional)
        the device the tensor will be stored to (default is 'cuda:0')

    Returns
    -------
    SparseTensor
        the sparse identity matrix
    """

    if cols is None:
        cols = rows
    n = min(rows, cols)
    return spdiag(torch.ones(n, dtype=dtype, device=device), rows=rows, cols=cols)


def speye_like(tensor):
    """
    Creates a sparse identity matrix with the same shape of the input tensor.

    Parameters
    ----------
    tensor : Tensor
        the input tensor

    Returns
    -------
    SparseTensor
        the sparse identity matrix
    """

    return speye(rows=row(tensor), cols=col(tensor), dtype=tensor.dtype, device=tensor.device)
