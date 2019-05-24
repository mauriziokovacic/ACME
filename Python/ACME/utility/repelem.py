import numpy as np
import torch
from .isnumpy     import *
from .istorch     import *
from .numpy2torch import *

def repelem(tensor,*size):
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
    if istorch(out):
        out = torch2numpy(out)
    if isnumpy(out):
        for d in range(0,len(size)):
            out = np.repeat(out,size[d],axis=d)
    if istorch(tensor):
        return numpy2torch(out,dtype=tensor.dtype,device=tensor.device)
    if isnumpy(tensor):
        return out
    assert False, 'Unknown data type'
