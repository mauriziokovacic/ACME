import torch
from functools import reduce


def barycenter(P, T=None, dim=0):
    """
    Returns the barycenters of the n-gons if a topology is provided, the
    barycenter of the input point set along the specified dimension otherwise

    Parameters
    ----------
    P : Tensor
        the point set tensor
    T : LongTensor (optional)
        the topology tensor (default is None)
    dim : int (optional)
        the dimension along the barycenter is computed (default is 0)

    Returns
    -------
    Tensor
        the barycenters tensor
    """

    if T is None:
        return torch.mean(P, dim, keepdim=True)
    p = tuple(P[T])
    return torch.div(reduce((lambda a, b: a + b), p), len(p))
