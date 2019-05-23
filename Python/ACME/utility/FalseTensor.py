import torch

def FalseTensor(*size,device='cuda:0'):
    """
    Returns a Tensor of type torch.uint8 containing only False values

    Parameters
    ----------
    *size : int
        the shape of the tensor
    device : str
        the device to store the tensor to

    Returns
    -------
    Tensor
        a uint8 precision tensor
    """

    return torch.zeros(*size,dtype=torch.uint8,device=device)


