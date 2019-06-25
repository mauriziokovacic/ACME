import torch
from ..utility.ConstantTensor import *


class ConstantLayer(torch.nn.Module):
    """
    Returns a constant tensor

    Attributes
    ----------
    value : float
        the value of the tensor
    dim : int...
        the shape of the tensor
    device : str or torch.device
        the device the tensor will be stored to

    Methods
    -------
    forward(input)
        returns a constant tensor
    """

    def __init__(self,value,*dim,device='cuda:0'):
        """
        Parameters
        ----------
        value : float
            the value of the tensor
        *dim : int...
            the shape of the tensor
        device : str or torch.device (optional)
            the device the tensor will be stored to (default is 'cuda:0')
        """

        super(ConstantLayer,self).__init__()
        self.value  = value
        self.dim    = dim
        self.device = device



    def forward(self,input):
        """
        Returns a constant tensor

        Parameters
        ----------
        input : Tensor
            the batch tensor

        Returns
        -------
        Tensor
            the constant tensor
        """

        return ConstantTensor(self.value,self.dim,device=self.device)



class ZerosLayer(ConstantLayer):
    """
    Returns a null tensor
    """

    def __init__(self,*dim,device='cuda:0'):
        """
        Parameters
        ----------
        *dim : int...
            the shape of the tensor
        device : str or torch.device (optional)
            the device the tensor will be stored to (default is 'cuda:0')
        """

        super(ZerosLayer,self).__init__(0,*dim,device=device)



class OnesLayer(ConstantLayer):
    """
    Returns a tensor of ones
    """

    def __init__(self,*dim,device='cuda:0'):
        """
        Parameters
        ----------
        *dim : int...
            the shape of the tensor
        device : str or torch.device (optional)
            the device the tensor will be stored to (default is 'cuda:0')
        """

        super(OnesLayer,self).__init__(1,*dim,device=device)
