import torch


def Uint8Tensor(values, device='cuda:0'):
    """
    Returns a Tensor of type torch.uint8 containing the given values

    Parameters
    ----------
    values : list
        the values of the tensor
    device : str
        the device to store the tensor to

    Returns
    -------
    Tensor
        a unsigned integer 8 precision tensor
    """

    return torch.tensor(values, dtype=torch.uint8, device=device)
