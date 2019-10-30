from .poisson_equation import *


def laplace_equation(L, size=1, eps=0.0001):
    """
    Solves the Laplace equation for the given Laplace matrix

    Parameters
    ----------
    L : Tensor or SparseTensor
        the (N,N,) Laplace matrix
    size : int
        the dimension of the known values (default is 1)
    eps : float (optional)
        a regularizer value (default is 0.0001)

    Returns
    -------
    Tensor
        the (N,) or (N,size,) solution tensor
    """

    return poisson_equation(L, torch.zeros(L.size(0), size, dtype=torch.float, device=L.device), eps=eps)
