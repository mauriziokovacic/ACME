import torch
from ..math.eye import *


def create_constraint_permutation(I, n, dtype=torch.float, device='cuda:0'):
    """
    Creates a permutation matrix that brings the given constraints to the bottom

    Parameters
    ----------
    I : LongTensor
        the constraint indices
    n : int
        the number of total constraints
    dtype : type (optional)
        the type of the output matrix (default is torch.float)
    device : str or torch.device (optional)
        the device the tensor will be stored to (default is 'cuda:0')

    Returns
    -------
    Tensor
        a permutation matrix
    """

    x    = torch.zeros(n, 1, dtype=torch.float, device=device)
    x[I] = 1
    x1   = torch.cumsum(1-x) * (1-x)
    x2   = (torch.max(x1)[0]+torch.cumsum(x)) * x
    P    = eye(n, dtype=dtype, device=device)#speye(N,N)
    P    = P[:, x1+x2]
    return P





