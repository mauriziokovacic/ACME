from .isnumpy import *
from .istorch import *


def diff(tensor, dim=-1):
    """
    Computes the element-wise differences along the specified dimension

    Parameters
    ----------
    tensor : Tensor
        the input tensor
    dim : int (optional)
        the dimension to compute the differences along

    Returns
    -------
    Tensor
        the differences tensor
    """

    if isnumpy(tensor):
        return numpy.diff(tensor, axis=dim)
    if istorch(tensor):
        i = torch.arange(0, tensor.shape[dim], dtype=torch.long, device=tensor.device)
        return torch.index_select(tensor, dim, i[1:]) - torch.index_select(tensor, dim, i[:-1])
    raise RuntimeError('Unknown data type. Expecting Numpy or PyTorch tensor, got {} instead'.format(tensor.__class__.__name__))