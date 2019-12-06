from ..utility.matmul import *
from .linear_problem  import *


def least_squares(A, b, eps=0.0001):
    """
    Solves a linear problem in the least squares sense:
    A^T Ax = A^T b

    Parameters
    ----------
    A : Tensor or SparseTensor
        the (N,M,) constraints matrix
    b : Tensor
        the (N,D,) known values tensor
    eps : float (optional)
        a regularizer value (default is 0.0001)

    Returns
    -------
    Tensor
        the (N,D,) solution tensor
    """

    return linear_problem(matmul(A.t(), A), matmul(A.t(), b), eps=eps)
