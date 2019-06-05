import torch

def onehot2index(onehot,dim=1):
    """
    Converts a onehot encoding into an indices tensor

    Parameters
    ----------
    onehot : Tensor
        a onehot encoding tensor
    dim : int (optional)
        the dimension along the encoding will be performed (default is 1)

    Returns
    -------
    LongTensor
        the indices tensor
    """

    return torch.argmax(onehot,dim,keepdim=True)
