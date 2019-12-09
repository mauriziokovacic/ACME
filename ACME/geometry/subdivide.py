from ..utility.static_vars  import *
from ..topology.subdivision import *
from .soup2mesh             import *


@static_vars(type={'tri': xtri,
                   'quad': xquad,
                   't2q': xtri2quad,
                   'tet': xtet,
                   'hex': xhex,
                   't2h': xtet2hex})
def subdivide(P, T, iter=1, type='tri'):
    """
    Subdivides the given mesh n times

    Parameters
    ----------
    P : Tensor
        the input points set
    T : LongTensor
        the topology tensor
    iter : int (optional)
        the number of times to subdivide the input mesh (default is 1)

    Returns
    -------
    (Tensor, LongTensor, Tensor)
        the new points set, the new topology and the subdivision matrix
    """

    if type in subdivide.type:
        fun = subdivide.type[type]
    else:
        raise RuntimeError('Unknown subdivision type. Please choose among:\n{}'.format('\n'.join(subdivide.type.keys())))
    M, t    = fun(T, iter=iter)
    p       = torch.matmul(M, P)
    p, t, I = soup2mesh(p, t)[0:3]
    return p, t, M[I]
