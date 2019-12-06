from ..math.knn           import *
from ..math.unrooted_norm import *


def normal_metric(Pi, Ni, Pj, Nj):
    """
    Returns the normal metric for two given oriented point sets

    Parameters
    ----------
    Pi : Tensor
        a (M,3,) points set tensor
    Ni : Tensor
        a (M,3,) normal tensor
    Pj : Tesnor
        a (N,3,) points set tensor
    Nj : Tensor
        a (M,3,) normal tensor

    Returns
    -------
    Tensor
        a (1,) metric tensor
    """

    i = knn(Pi, Pj, 1, distFcn=sqdistance)[0].squeeze()
    j = knn(Pj, Pi, 1, distFcn=sqdistance)[0].squeeze()
    return torch.sum(sqdistance(Ni, Nj[i,:], dim=1)) + \
           torch.sum(sqdistance(Nj, Ni[j,:], dim=1))
