from ACME.utility.row     import *
from ACME.utility.col     import *
from ACME.utility.ndim    import *
from ACME.utility.reshape import *
from ACME.utility.flatten import *

def affine2linear(M):
    """
    Returns the flattened version of the input affine matrix, without the last row

    Parameters
    ----------
    M : Tensor
        a (M,4,4,) or (M,3,3,) tensor

    Returns
    -------
    Tensor
        the (M,12) or (M,6) tensor
    """

    return flatten(M)[:-col(M)] if ndim(M)==2 else reshape(M,(row(M),-1))
