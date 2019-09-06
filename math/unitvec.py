import torch


def unitvec(size, i, dtype=torch.float, device='cuda:0'):
    """
    Creates a 1xn zero tensor with a single 1 in the specified position

    Parameters
    ----------
    size : int
        number of elements in the tensor
    i : int
        position of the 1 in the tensor
    dtype : type (optional)
        type of the tensor (default is torch.float)
    device : str or torch.device (optional)
        device where to store the tensor to (default is 'cuda:0')

    Returns
    -------
    Tensor
        a 1xn tensor
    """

    e = torch.zeros(size, dtype=dtype, device=device)
    e[i] = 1
    return e.unsqueeze(0)
