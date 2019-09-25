import numpy
import torch
from .flatten    import *
from .islist     import *
from .istuple    import *
from .isscalar   import *
from .istensor   import *
from .isnumpy    import *
from .istorch    import *


def bitshift(obj, k):
    """
    Performs a bitshift opeartion over the input object

    Parameters
    ----------
    obj : scalar or list or tuple or Tensor
        the input object to perform the bitshift on
    k : int
        the number of shift to perform. Positive values perform a left shift, negative values a right shift

    Returns
    -------
    scalar or list or tuple or Tensor
        the bitshifted object
    """

    if istensor(obj):
        if isnumpy(obj):
            x = [value << k if k > 0 else value >> -k for value in obj]
            return numpy.array(x)
        if istorch(obj):
            x = [value << k if k > 0 else value >> -k for value in obj.view(-1)]
            return torch.tensor(x, dtype=obj.dtype, device=obj.device).view(*obj.shape)

    if istuple(obj) or islist(obj):
        return [value << k if k > 0 else value >> -k for value in obj]

    if isscalar(obj):
        return obj << k if k > 0 else obj >> -k

    raise RuntimeError('Unknown data type')
