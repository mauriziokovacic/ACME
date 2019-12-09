from ..utility.SparseTensor import *


def spzeros(size, device='cuda:0'):
    """
    Returns a zero sparse tensor with the specified size

    Parameters
    ----------
    size : tuple
        the size of the sparse tensor
    device : str or torch.device (optional)
        the device to store the tensors to (default is 'cuda:0')

    Returns
    -------
    SparseTensor
        an empty sparse tensor of the specified size
    """

    return SparseTensor(size=size).to(device=device)
