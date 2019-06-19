import torch.sparse

def issparse(*obj):
    """
    Returns whether or not the inputs are PyTorch sparse tensors

    Parameters
    ----------
    *obj : object...
        a sequence of objects

    Returns
    -------
    bool
        True if the inputs are PyTorch sparse Tensors, False otherwise
    """

    return all([isinstance(o,torch.sparse.FloatTensor) for o in obj])
