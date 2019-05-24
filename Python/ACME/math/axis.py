import torch
from .unitvec import *

def x_axis(dim=3,dtype=torch.float,device='cuda:0'):
    """
    Creates a 1xn tensor in the form [1,0,...,0]

    Parameters
    ----------
    dim : int (optional)
        number of elements in the tensor (defaul is 3)
    dtype : type (optional)
        type of the tensor (default is torch.float)
    device : str or torch.device (optional)
        device where to store the tensor to (default is 'cuda:0')

    Returns
    -------
    Tensor
        a 1xn tensor

    Raises
    ------
    AssertError
        if dim is less than 1
    """

    assert dim>=0, 'Dimension should be at least 1.'
    return unitvec(dim,0,dtype=dtype,device=device)



def y_axis(dim=3,dtype=torch.float,device='cuda:0'):
    """
    Creates a 1xn tensor in the form [0,1,0,...,0]

    Parameters
    ----------
    dim : int (optional)
        number of elements in the tensor (defaul is 3)
    dtype : type (optional)
        type of the tensor (default is torch.float)
    device : str or torch.device (optional)
        device where to store the tensor to (default is 'cuda:0')

    Returns
    -------
    Tensor
        a 1xn tensor

    Raises
    ------
    AssertError
        if dim is less than 2
    """

    assert dim>=2, 'Dimension should be at least 2.'
    return unitvec(dim,1,dtype=dtype,device=device)



def z_axis(dim=3,dtype=torch.float,device='cuda:0'):
    """
    Creates a 1xn tensor in the form [0,0,1,0...,0]

    Parameters
    ----------
    dim : int (optional)
        number of elements in the tensor (defaul is 3)
    dtype : type (optional)
        type of the tensor (default is torch.float)
    device : str or torch.device (optional)
        device where to store the tensor to (default is 'cuda:0')

    Returns
    -------
    Tensor
        a 1xn tensor

    Raises
    ------
    AssertError
        if dim is less than 3
    """

    assert dim>=3, 'Dimension should be at least 3.'
    return unitvec(dim,2,dtype=dtype,device=device)



def w_axis(dim=4,dtype=torch.float,device='cuda:0'):
    """
    Creates a 1xn tensor in the form [0,0,0,1,0,...,0]

    Parameters
    ----------
    dim : int (optional)
        number of elements in the tensor (defaul is 4)
    dtype : type (optional)
        type of the tensor (default is torch.float)
    device : str or torch.device (optional)
        device where to store the tensor to (default is 'cuda:0')

    Returns
    -------
    Tensor
        a 1xn tensor

    Raises
    ------
    AssertError
        if dim is less than 4

    """

    assert dim>=4, 'Dimension should be at least 4.'
    return unitvec(dim,3,dtype=dtype,device=device)
