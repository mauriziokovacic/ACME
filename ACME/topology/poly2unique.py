import torch
from ..utility.unique import *
from .polysort        import *

def poly2unique(T,winding=False):
    """
    Returns the unique n-gons in the topology

    Parameters
    ----------
    T : LongTensor
        the topology tensor
    winding : bool (optional)
        if True takes into account winding ordering (default is False)

    Returns
    -------
    (LongTensor,LongTensor,LongTensor)
        returns the unique n-gons, their indices within the input tensor and the input tensor indices within the unique
    """

    C,ia,ic = unique(torch.t(polysort(T,winding=winding)),ByRows=True)
    return torch.t(C),ia,ic
