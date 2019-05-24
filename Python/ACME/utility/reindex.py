from .unique  import *
from .indices import *
from .squeeze import *

def reindex(I):
    """
    Retargets the input n indices in the range [0-(n-1)]

    Parameters
    ----------
    I : LongTensor
        the input indices

    Returns
    -------
    (LongTensor,LongTensor,LongTensor)
        the retargeted indices tensor, the unique input indices, the retargeted unique indices
    """

    i    = unique(I)
    j    = squeeze(indices(0,numel(i),device=I.device))
    x    = torch.zeros(i.max()+1,dtype=torch.long,device=I.device)
    x[i] = j
    ind  = x[I]
    return ind,i,j

