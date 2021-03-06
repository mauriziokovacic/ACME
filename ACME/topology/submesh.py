import torch
from ..utility.col     import *
from ..utility.isempty import *
from ..utility.strcmpi import *
from ..utility.indices import *
from ..utility.numel   import *
from ..utility.unique  import *
from ..utility.find    import *


def submesh(T, ind, type='node'):
    """
    Given a set of a mesh nodes/polygons indices, create the indices for constructing a submesh.

    Parameters
    ----------
    T : LongTensor
        the topology tensor
    ind : LongTensor
        a set of indices representing the nodes or the polygons to keep
    type : str (optional)
        the entity the input indices are referring to. It has to be either 'node' or 'topology'

    Returns
    -------
    (LongTensor,LongTensor,LongTensor)
        the indices of the polygons, the indices of the nodes and the new topology tensor
    """

    n = torch.max(T)[0]+1
    if not isempty(ind):
        k  = torch.zeros(n, 1, dtype=torch.long, device=T.device)
        if strcmpi(type, 'node'):
            In     = ind
        if strcmpi(type, 'topology'):
            It     = ind
            In     = unique(T[:, It])[0]
        k[In] = indices(1, numel(In))
        t     = k[T]
        if strcmpi(type, 'node'):
            j  = find(t > 0)[1]
            It = j
        if strcmpi(type, 'topology'):
            j = find(torch.prod(t, 0) > 0)[1]
        t = t[:, j]-1
    return It, In, t
