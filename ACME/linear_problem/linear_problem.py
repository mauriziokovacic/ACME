from ..math.speye      import *
from ..utility.isdense import *


def linear_problem(A, b, eps=0.0001):
    """
    Solves the linear problem Ax=b

    Parameters
    ----------
    A : Tensor or SparseTensor
        the (N,M,) constraints matrix
    b : Tensor
        the (N,D,) known values tensor
    eps : float (optional)
        a regularizer scalar value (default is 0.0001)

    Returns
    -------
    Tensor
        the (N,D,) solution tensor
    """

    M = A + eps * speye_like(A)
    return torch.solve(b if b.ndimension() >= 2 else torch.unsqueeze(b, -1),
                       M if isdense(M) else M.to_dense())[0].squeeze()