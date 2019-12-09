from ..utility.row import *
from ..utility.col import *
from .diag         import *


def eye(rows, cols=None, batch=1, dtype=torch.float, device='cuda:0'):
    """
    Creates an identity matrix with the specified shape. If columns are not specified
    the matrix is intended squared.

    Parameters
    ----------
    rows : int
        the rows of the matrix
    cols : int (optional)
        the columns of the matrix
    batch : int (optional)
        the number of matrices to generate (default is 1)
    dtype : type (optional)
        the type of the values (default is torch.float)
    device : str or torch.device (optional)
        the device the tensor will be stored to (default is 'cuda:0')

    Returns
    -------
    Tensor
        the (N,M,) identity matrix
    """

    if cols is None:
        cols = rows
    n = min(rows, cols)
    return torch.cat([diag(torch.ones(n, dtype=dtype, device=device),
                           rows=rows, cols=cols).unsqueeze(0) for i in range(batch)], dim=0).squeeze()


def eye_like(tensor):
    """
    Creates an identity matrix with the same shape of the input tensor.

    Parameters
    ----------
    tensor : Tensor
        the input tensor

    Returns
    -------
    Tensor
        the identity matrix
    """

    return eye(rows=row(tensor), cols=col(tensor), dtype=tensor.dtype, device=tensor.device)
