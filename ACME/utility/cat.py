from .isdense  import *
from .issparse import *


def cat(tensors, *args, **kwargs):
    """
    Concatenates the given tensors

    Parameters
    ----------
    tensors : tuple or list
        a series of tensors
    args : ...
    kwargs : ...

    Returns
    -------
    Tensor
        the concatenation tensor
    """

    if issparse(*tensors) or isdense(*tensors):
        return torch.cat(tensors, *args, **kwargs)
    return torch.cat([t.to_dense() if issparse(t) else t for t in tensors], *args, **kwargs)
