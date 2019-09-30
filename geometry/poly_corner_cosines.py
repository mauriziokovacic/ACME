from ..math.acos    import *
from ..math.dot     import *
from ..math.normvec import *
from .poly_edges    import *


def poly_corner_cosines(P, T):
    """
    Returns the per polygon corner cosines

    Parameters
    ----------
    P : Tensor
        a (N, D,) points set tensor
    T : LongTensor
        a (M, T,) topology tensor

    Returns
    -------
    Tensor
        the (T, M,) corner cosines tensor
    """

    E = torch.cat([normr(e).unsqueeze(0) for e in poly_edges(P, T)], dim=0)
    E = torch.cat((E[-1].unsqueeze(0), E), dim=0)
    return dot(E[1:], -E[:-1], dim=2).squeeze().t()
