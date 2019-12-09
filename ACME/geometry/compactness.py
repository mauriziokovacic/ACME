from ..geometry.barycenter import *
from ..math.unrooted_norm  import *


def compactness(P):
    """
    Measures the compactness of the given points set

    Parameters
    ----------
    P : Tensor
        a (...,N,D,) points set tensor

    Returns
    -------
    Tensor
        a (...,N,) compactness tensor
    """

    return torch.std(
        sqdistance(P, barycenter(P, dim=1)).squeeze(-1),
        dim=-1,
    )
