import torch
from utility.numel      import *
from utility.LongTensor import *

def diag(tensor,rows=None,cols=None):
    """
    Creates a diagonal matrix from the given input tensor

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
    Tensor
        a diagonal matrix with the given entries on the diagonal
    """

    if rows is None:
        rows = numel(tensor)
    if cols is None:
        cols = rows
    n = min(rows,cols)
    v = torch.cat((tensor.flatten(),torch.zeros(n-numel(tensor),dtype=tensor.dtype,device=tensor.device)))
    return torch.diag(v)
