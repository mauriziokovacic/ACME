from ..math.cross   import *
from ..math.normvec import *
from .poly_edges    import *


def poly_corner_normals(P, T):
    """
    Returns the per polygon corner normals

    Parameters
    ----------
    P : Tensor
        a (N, D,) points set tensor
    T : LongTensor
        a (M, T,) topology tensor

    Returns
    -------
    tuple
        M (N, D,) corner normal tensors
    """

    E = torch.cat([normr(e).unsqueeze(0) for e in poly_edges(P, T)], dim=0)
    E = torch.cat((E[-1].unsqueeze(0), E), dim=0)
    return tuple(cross(E[1:], -E[:-1], dim=2))
