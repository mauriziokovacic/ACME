import torch
from .unitvec import *

def rotm2affine(M):
    """
    Returns the affine matrix from the input rotation matrix

    The affine matrix is represented as:
       A = |M 0|
           |0 1|
    where 0 is a zero row/column vector and 1 is a single value

    Parameters
    ----------
    M : Tensor
        a (3,3) or (2,2) matrix

    Returns
    -------
    Tensor
        a (4,4) or (3,3) matrix
    """

    return torch.cat((torch.cat((M,torch.zeros(row(M),1,dtype=M.dtype,device=M.device)),dim=1),
                      unitvec(col(M)+1,col(M),dtype=M.dtype,device=M.device)),dim=0)
