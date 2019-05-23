import torch
import torch_scatter

def accumarray(I,V,size=None,dim=0):
    """
    Returns a Tensor by accumulating elements of tensor V using the subscripts I

    Parameters
    ----------
    I : LongTensor
        the indices of the tensor
    V : Tensor
        the values of the tensor
    size : (optional)
    dim : (optional)

    Returns
    -------
    Tensor
        a tensor formed by accumulating the input values in the respective input indices positions
    """

    if size is None:
        size = torch.max(I).item()+1
    return torch_scatter.scatter_add(V,I,dim=dim,dim_size=size)
