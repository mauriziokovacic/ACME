import torch

def bounding_box(P,dim=0):
    """
    Returns the bounding box minimum and maximum points of the given
    input point set along the specified dimension

    Parameters
    ----------
    P : Tensor
        the input point set
    dim : int (optional)
        the dimension along the boundng box is computed (default is 0)

    Returns
    -------
    (Tensor,Tensor)
        the minimum and maximum points of the input bounding box
    """

    return torch.min(P,dim,keepdim=True)[0],torch.max(P,dim,keepdim=True)[0]
