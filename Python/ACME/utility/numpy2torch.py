import numpy
import torch
from .isnumpy import *
from .istorch import *

def numpy2torch(tensor,dtype=torch.float,device='cuda:0'):
    """
    Converts the input tensor from PyTorch to Numpy.

    Parameters
    ----------
    tensor : Tensor
        the input tensor
    dtype : type (optional)
        the type of the output tensor (default is torch.float)
    device : str or torch.device (optional)
        the device the tensor will be stored to (default is 'cuda:0')

    Returns
    -------
    Tensor
        the converted tensor

    Raises
    ------
    AssertionError
        if input tensor is neither a Numpy or PyTorch tensor
    """

    if istorch(tensor):
        return tensor
    if isnumpy(tensor):
        return torch.from_numpy(out).to(dtype=dtype,device=device)
    assert False, 'Unknown data type'
