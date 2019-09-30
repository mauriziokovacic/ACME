from ..math.cot     import *
from ..math.normvec import *
from .poly_edges    import *


def poly_corner_cotangents(P, T):
    """
    Returns the per polygon corner otangents

    Parameters
    ----------
    P : Tensor
        a (N, D,) points set tensor
    T : LongTensor
        a (M, T,) topology tensor

    Returns
    -------
    Tensor
        the (T, M,) corner cotangents tensor
    """

    E = torch.cat([normr(e).unsqueeze(0) for e in poly_edges(P, T)], dim=0)
    E = torch.cat((E[-1].unsqueeze(0), E), dim=0)
    return cot(E[1:], -E[:-1], dim=2).squeeze().t()
