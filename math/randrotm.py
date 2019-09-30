from .eul2rotm import *


def randrotm(n=1, device='cuda:0'):
    """
    Returns n randorm rotation matrices

    Parameters
    ----------
    n : int (optional)
        the number of rotation matrices to generate (default is 1)
    device : str (optional)
        the device to store the tensor to (default is 'cuda:0')

    Returns
    -------
    Tensor
        the (n,3,3,) rotation matrices tensor
    """

    return eul2rotm(torch.rand(n, 3, dtype=torch.float, device=device))
