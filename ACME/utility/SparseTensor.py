import torch.sparse

def SparseTensor(size=None,indices=None,values=None):
    """
    Returns a sparse.FloatTensor containing the given values

    Parameters
    ----------
    size : tuple (optional)
        the output tensor size. If None it will be automatically computed (default is None)
    indices : LongTensor (optional)
        the indices tensor (default is None)
    values : Tensor (optional)
        the values tensor (default is None)

    Returns
    -------
    SparseTensor
        a sparse tensor

    Raises
    ------
    AssertionError
        if indices and values are not both valid or None
    """

    assert !((indices is None)!=(values is None)), 'Indices and values must be either both valid or None.'

    if size is None:
        if indices is None:
            return torch.sparse.FloatTensor()
        return torch.sparse.FloatTensor(indices,values)
    if indices is None:
        return torch.sparse.FloatTensor(*size)
    return torch.sparse.FloatTensor(indices,values,size)

