import torch
from ACME.utility.row     import *
from ACME.utility.col     import *
from ACME.utility.ndim    import *
from ACME.utility.reshape import *
from ACME.utility.flatten import *

def linear2affine(M):
    """
    Returns a stack of affine matrices from the input linearized matrices

    The input matrices are intended without the last row

    Parameters
    ----------
    M : Tensor
        a (M,6,) or (M,12,) tensor of linearized matrices

    Returns
    -------
    Tensor
        the (M,3,3,) or (M,4,4,) matrices tensor
    """

    return reshape(
                torc.cat((M,
                          torch.zeros(row(M),2 if col(M)==6 else 3,dtype=M.dtype,device=M.device),
                          torch.ones( row(M),1,dtype=M.dtype,device=M.device)),dim=1),
                (row(M),3 if col(M)==6 else 4,-1))
