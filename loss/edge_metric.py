from ..math.unrooted_norm import *
from ..topology.poly2edge import *


def edge_metric(P, T):
    """
    Returns the edge metric for the given points set and its topology

    Parameters
    ----------
    P : Tensor
        the points set tensor
    T : LongTensor
        the topology tensor

    Returns
    -------
    Tensor
        the (1,) metric tensor
    """

    Pi, Pj = P[poly2edge(T)[0]]
    return torch.mean(sqdistance(Pi, Pj, dim=1))
