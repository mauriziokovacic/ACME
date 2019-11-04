import numpy
import torch
from .isnumpy     import *
from .istorch     import *
from .numpy2torch import *
from .torch2numpy import *


def repelem(tensor, *size):
    """
    Repeats the tensor values along the tensor dimensions by the given times

    Example:
        repelem([[1,2,3]],1,2) -> [[1,1,2,2,3,3]]

    Parameters
    ----------
    tensor : Tensor
        the input tensor
    *size : int...
        a sequence of times to repeats the tensor values along a particular dimension

    Returns
    -------
    Tensor
        a tensor
    """

    out = tensor
    if isnumpy(tensor):
        for d in range(0, len(size)):
            out = numpy.repeat(out, size[d], axis=d)
        return out
    if istorch(tensor):
        for d in range(0, len(size)):
            out = torch.repeat_interleave(out, size[d], dim=d)
        return out
    assert False, 'Unknown data type'
