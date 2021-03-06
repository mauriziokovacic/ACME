import torch
from ..utility.row     import *
from ..utility.col     import *
from ..utility.reindex import *
from ..utility.unique  import *
from ..topology.ispoly import *


def unsubdivide(P, T, iter=1):
    """
    Unsubdivides the given mesh n times

    In order to work, the mesh is intended subdivided using the method 'subdivide'.

    Parameters
    ----------
    P : Tensor
        the input points set
    T : LongTensor
        the topology tensor
    iter : int (optional)
        the number of times to unsubdivide the input mesh (default is 1)

    Returns
    -------
    (Tensor,LongTensor,LongTensor)
        the new points set, the new topology, and the indices of the original points
    """

    i = T.clone()
    for n in range(0, iter):
        i = torch.t(torch.reshape(i[0], (col(i)//4, 4)))
        if istri(T):
            i = i[0:3]
        if row(i) == 1:
            i = torch.t(i)
    j = unique(i)[0]
    p = P[j]
    t = reindex(i)
    i = j
    return p, t, i
