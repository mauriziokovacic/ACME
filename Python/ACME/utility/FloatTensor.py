import torch

def FloatTensor(values,device='cuda:0'):
    """
    Returns a Tensor of type torch.float containing the given values

    Parameters
    ----------
    values : list
        the values of the tensor
    device : str
        the device to store the tensor to

    Returns
    -------
    Tensor
        a floating point precision tensor
    """

    return torch.tensor(values,dtype=torch.float,device=device)
