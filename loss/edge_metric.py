from ..math.unrooted_norm  import *
from ..geometry.poly_edges import *


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

    return poly_edges_mean_length(P, T, distFcn=sqnorm)
