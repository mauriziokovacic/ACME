from ACME.utility.col     import *
from ACME.utility.flatten import *

def affine2linear(M):
    """
    Returns the flattened version of the input affine matrix, without the last row

    Parameters
    ----------
    M : Tensor
        a (4,4) or (3,3) tensor

    Returns
    -------
    Tensor
        a (1,12) or (1,6) tensor
    """

    return flatten(M)[:-col(M)]
