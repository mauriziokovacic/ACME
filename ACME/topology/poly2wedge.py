from ..utility.col        import *
from ..utility.row        import *
from ..utility.indices    import *
from ..utility.circrepeat import *
from ..utility.repmat     import *


def poly2wedge(T):
    """
    Returns the wedge tensor of the given topology

    A wedge is a triplet of consecutive nodes within a poly

    Parameters
    ----------
    T : LongTensor
        the topology tensor

    Returns
    -------
    (LongTensor, LongTensor)
        the wedge tensor and the respective polygon indices
    """

    n = row(T)
    i = indices(0, n-1, device=T.device)
    i = torch.cat((i, (i+1) % n, (i+2) % n), dim=1)
    return torch.cat(tuple(T[i]), dim=1), repmat(indices(0, col(T)-1, device=T.device), n, 1).squeeze()
