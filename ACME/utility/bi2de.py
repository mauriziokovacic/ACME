import numpy
import torch
from .ndim     import *
from .reshape  import *
from .isint    import *
from .isstring import *
from .istensor import *
from .isnumpy  import *
from .istorch  import *

def bi2de(obj, dim=-1):
    """
    Converts a binary number into its decimal representation

    Parameters
    ----------
    obj : int or str or Tensor
        A number in binary format or a uint8 Numpy or PyTorch tensor
    dim : -1 (optional)
        if obj is a tensor, the dimension along the convertions is performed

    Returns
    -------
    int or Tensor
        the decimal representation of the input number or tensor dimension

    Raises
    ------
    AssertionError
        if data type is unknown
    """

    if isint(obj) or isstring(obj):
        return int(str(obj), 2)
    if istensor(obj):
        s      = [1, ]*ndim(obj)
        s[dim] = size(obj)[dim]
        if isnumpy(obj):
            i      = numpy.array([2**i for i in range(size(obj)[dim]-1, -1, -1)])
        if istorch(obj):
            i      = torch.tensor([2**i for i in range(size(obj)[dim]-1, -1, -1)], dtype=torch.uint8, device=obj.device)
        i = reshape(i,s)
        return sum(obj*i, dim, keepdim=True)
    assert False, 'Unknown data type'
