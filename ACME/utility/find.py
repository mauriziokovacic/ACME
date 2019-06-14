import torch
from .numel   import *
from .ndim    import *
from .indices import *
from .flatten import *

def find(cond,linear=True):
    """
    Finds the indices of the True values within the given condition

    Parameters
    ----------
    cond : uint8 Tensor
        a tensor derived from a condition (Ex.: t<0)
    linear : bool (optional)
        a flag driving the indices extraction. True returns the linear indices, False returns the subscripts (default is True)

    Returns
    -------
    LongTensor
        a list of indices or a (ndim(cond),true values,) tensor containing the subscripts
    """

    return torch.nonzero(cond.flatten()).flatten() if linear else torch.t(torch.nonzero(cond))
