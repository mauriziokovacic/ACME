import torch

def IntTensor(values,device='cuda:0'):
    """
    Returns a Tensor of type torch.int containing the given values

    Parameters
    ----------
    values : list
        the values of the tensor
    device : str
        the device to store the tensor to

    Returns
    -------
    Tensor
        an integer precision tensor
    """

    return torch.tensor(values,dtype=torch.int,device=device)
