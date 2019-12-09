import torch


def linspace(min, max, n, dtype=torch.float, device='cuda:0'):
    """
    Generates a tensor of n evenly spaced values between min and max

    Example:
        linspace(0,1,11,device='cpu') -> torch.tensor([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],dtype=torch.float,device='cpu')

    Parameters
    ----------
    min : int or float
        the minimum value
    max : int or float
        the maximum value
    n : int
        the number of values to generate
    dtype : type (optional)
        the type of the tensor (default is torch.float)
    device : str or torch.device (optional)
        the device to store the tensor to (default is 'cuda:0')

    Returns
    -------
    Tensor
        a tensor with n evenly spaced values between min and max
    """

    return torch.linspace(min, max, n).to(dtype=dtype, device=device).unsqueeze(1)
