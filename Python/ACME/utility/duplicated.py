from .row     import *
from .indices import *
from .setdiff import *
from .unique  import *

def duplicated(V,ByRows=False):
    """
    Returns the duplicated values within the input tensor and their indices

    Parameters
    ----------
    V : LongTensor
        the input tensor
    ByRows : bool (optional)
        if True treats the rows as a single entity

    Returns
    -------
    (LongTensor,LongTensor)
        returns the tensor of duplicated values/rows, their indices within the input tensor
    """

    i = indices(0,row(V)-1,device=V.device)
    I = unique(V,ByRows=ByRows)[1]
    I = setdiff(i,I)
    return V[I],I
