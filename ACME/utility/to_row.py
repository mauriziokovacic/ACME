from .to_fat  import *
from .flatten import *

def to_row(*tensors):
    """
    Transforms the input tensor into a row vector

    If the input tensor is a matrix, it will be flattened into a (1,n) vector
    If the input tensor is a (n,) vector, it will be unsqueezed in dimension 0
    No operation is performed if the input is already a (1,n) vector

    Parameters
    ----------
    *tensors : Tensor...
        a sequence of (n,) or (n,m) tensors

    Returns
    -------
    Tensor or list
        a (1,n) or (m,n) tensor, input if input is (1,n)

    Raises
    ------
    AssertError
        if inputs are not tensors
    """

    out = [to_fat(flatten(t)) for t in tensors]
    return out if len(out)>1 else out[0]
