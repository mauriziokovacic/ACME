import torch
from ACME.math.unitvec import *

def index2onehot(index,n=None):
    """
    Converts an input indices tensor into a onehot encoding of n classes

    Parameters
    ----------
    index : LongTensor
        the input indices tensor
    n : int (optional)
        the number of classes. If None it will be automatically set to max(index) (default is None)

    Returns
    -------
    Tensor
        the onehot encoding tensor
    """

    if n is None:
        n = torch.max(index)[0].item()
    return torch.cat(tuple(unitvec(n,x.item(),dtype=torch.float,device=index.device) for x in index),dim=0)
