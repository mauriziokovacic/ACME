from ACME.utility.row        import *
from ACME.utility.indices    import *
from ACME.utility.circrepeat import *

def poly2wedge(T):
    """
    Returns the wedge tensor of the given topology

    A wedge is a triplet of consecutive nodes

    Parameters
    ----------
    T : LongTensor
        the topology tensor

    Returns
    -------
    LongTensor
        the wedge tensor
    """

    n = row(T)
    i = indices(0,n-1,device=T.device)
    i = torch.cat((i,(i+1)%n,(i+2)%n),dim=0)
    return torch.cat(tuple(T[i]),dim=1)
