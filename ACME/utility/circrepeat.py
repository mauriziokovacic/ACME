import numpy
import torch
from .isnumpy     import *
from .istorch     import *
from .ndim        import *
from .numpy2torch import *
from .torch2numpy import *


def circrepeat(tensor, k, dim=1):
    """
    Repeats the value of the tensor along the specified dimensions

    Parameters
    ----------
    tensor : Tensor
        the input tensor

    *n : int
        number of repetitions along each dimension

    Returns
    -------
    Tensor
        the tensor with repeated values

    Raises
    ------
    AssertionError
        if input tensor is neither a Numpy or PyTorch tensor
    """

    out = tensor
    if istorch(out):
        out = torch2numpy(out)
    if isnumpy(out):
        out = numpy.pad(tensor, (tuple([(0, 0 if i != dim else k) for i in range(0, ndim(out))])), mode='wrap')
    if istorch(tensor):
        return numpy2torch(out, dtype=tensor.dtype, device=tensor.device)
    if isnumpy(tensor):
        return out
    assert False, 'Unknown data type'
