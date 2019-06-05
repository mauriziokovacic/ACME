from .transpose import *
from .matmul    import *

def to_sym_square(*tensors):
    """
    Returns the symmetric square matrix given by multiplying a tensor for its transpose

    Parameters
    ----------
    *tensors : Tensor...
         a sequence of tensors

    Returns
    -------
    Tensor or list
        the symmetric square matrix of the input tensor

    Raises
    ------
    AssertError
        if input is not a tensor
    """

    out = [matmult(t,transpose(t)) for t in tensors]
    return out if len(out)>1 else out[0]
