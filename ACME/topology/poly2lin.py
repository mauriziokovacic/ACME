from ..utility.row     import *
from ..utility.col     import *
from ..utility.flatten import *
from ..utility.indices import *
from ..utility.repmat  import *


def poly2lin(T):
    """
    Returns the indices of the topology nodes, along with their
    respective polygon indices.

    Parameters
    ----------
    T : LongTensor
        the topology tensor

    Returns
    -------
    (LongTensor,LongTensor)
        the nodes indices and the respective polygon indices
    """

    return flatten(T), repmat(indices(0, col(T)-1), row(T), 1)
