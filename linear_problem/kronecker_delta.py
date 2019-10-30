import torch


def kronecker_delta(n, i, device='cuda:0'):
    """
    Builds the Kronecker delta with the given shape

    Parameters
    ----------
    n : int
        the number of elements in the tensor
    i : LongTensor
        the index of the 1s in the tensor
    device : str or torch.device (optional)
        the device to store the tensor to (default is 'cuda:0')

    Returns
    -------
    Tensor
        the (N,) Kronecker delta tensor
    """

    k = torch.zeros(n, dtype=torch.float, device=device)
    k[i] = 1
    return k
