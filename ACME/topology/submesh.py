import torch
from ACME.utility.col     import *
from ACME.utility.isempty import *
from ACME.utility.strcmpi import *
from ACME.utility.indices import *
from ACME.utility.numel   import *
from ACME.utility.unique  import *
from ACME.utility.find    import *

def submesh(T,ind,type='node'):
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
        k  = torch.zeros(n,1,dtype=torch.long,device=T.device)
        if strcmpi(type,'node'):
            In     = ind
        if strcmpi(type,'topology'):
            It     = ind
            In     = unique(T[:,It])[0]
        k[In] = indices(1,numel(In))
        t     = k[T]
        if strcmpi(type,'node'):
            _,j = find(t>0)
            It  = j
        if strcmpi(type,'topology'):
            _,j = find(torch.prod(t,0)>0)
        t = t[:,j]-1
    return It,In,t
