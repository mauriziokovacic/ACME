import numpy
import torch
from .isnumpy  import *
from .istorch  import *
from .issparse import *


def sum(tensor, dim=-1, keepdim=True, **kwargs):
    """
    Returns the sum of the tensor elements along the specified dimension

    Parameters
    ----------
    tensor : Tensor
        a Numpy or PyTorch tensor
    dim : int (optional)
        the dimension along the sum is computed (default is -1)
    keepdim : bool (optional)
        if True the dimensions of the tensor are kept (default is True)
    **kwargs : ...
        any other argument the respective Numpy and PyTorch functions take

    Returns
    -------
    Tensor
        the tensor containing the sum of the input tensor elements
    """

    if isnumpy(tensor):
        return numpy.sum(tensor, axis=dim, keepdims=keepdim, **kwargs)
    if istorch(tensor):
        if issparse(tensor):
            return torch.sum(tensor.to_dense(), dim, keepdim=keepdim, **kwargs)
        return torch.sum(tensor, dim, keepdim=keepdim, **kwargs)
    assert False, 'Unknown data type'
