from . import linspace

def indices(min,max,dtype=torch.long,device='cuda:0'):
    """
    Generates a tensor of indices between min and max

    Example:
        indices(0,10,device='cpu') -> torch.tensor([0,1,2,3,4,5,6,7,8,9,10],dtype=torch.long,device='cpu')

    Parameters
    ----------
    min : int
        the minimum value
    max : int
        the maximum value
    dtype : type (optional)
        the type of the tensor (default is torch.long)
    device : str or torch.device (optional)
        the device to store the tensor to (default is 'cuda:0')

    Returns
    -------
    Tensor
        a tensor with values between min and max
    """

    return linspace(min,max,abs(min-max)+1,dtype=dtype,device=device)
