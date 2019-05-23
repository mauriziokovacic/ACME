import torch

def LongTensor(values,device='cuda:0'):
    """
    Returns a Tensor of type torch.long containing the given values

    Parameters
    ----------
    values : list
        the values of the tensor
    device : str
        the device to store the tensor to

    Returns
    -------
    Tensor
        a long integer precision tensor
    """

    return torch.tensor(values,dtype=torch.long,device=device)
