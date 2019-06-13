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
    LongTensor or (LongTensor,LongTensor)
        a list of indices or a two-dimensional tensor containing the subscripts
    """

    if linear or ndim(cond.squeeze())==1:
        i = indices(0,numel(cond)-1,device=cond.device)
        c = flatten(cond)
        return i[c]
    i = tuple(indices(0,d-1,device=cond.device).squeeze() for d in cond.shape)
    i = torch.meshgrid(*i)
    i = tuple(flatten(torch.t(x.to(device=cond.device)) for x in i)
    c = flatten(cond)
    i = tuple(j[c] for j in i)
    return i

    #return torch.nonzero(cond.flatten()).flatten() if linear else torch.nonzero(cond)
