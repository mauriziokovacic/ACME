import torch
from ..color.fetch_texture import *


class Sampler1D(torch.nn.Module):
    """
    A class representing a data sampler

    Methods
    -------
    forward(input,param)
        returns the data read from the given input with the given parametrization
    """

    def __init__(self):
        super(Sampler1D,self).__init__()

    def forward(self,input,param):
        """
        Returns the data read from the given input with the given parametrization

        Parameters
        ----------
        input : Tensor
            the input (N,C,) tensor
        param : Tensor
            the given (U,) parametrization tensor

        Returns
        -------
        Tensor
            the sampled (U,C,) tensor
        """

        return fetch_texture1D(input,param,mode='bilinear')



class Sampler2D(torch.nn.Module):
    """
    A class representing a data sampler

    Methods
    -------
    forward(input,param)
        returns the data read from the given input with the given parametrization
    """

    def __init__(self):
        super(Sampler2D,self).__init__()

    def forward(self,input,param):
        """
        Returns the data read from the given input with the given parametrization

        Parameters
        ----------
        input : Tensor
            the input (C,W,H) tensor
        param : Tensor
            the given (UV,2,) parametrization tensor

        Returns
        -------
        Tensor
            the sampled (UV,C,) tensor
        """

        return fetch_texture2D(input,param,mode='bilinear')



class Sampler3D(torch.nn.Module):
    """
    A class representing a data sampler

    Methods
    -------
    forward(input,param)
        returns the data read from the given input with the given parametrization
    """

    def __init__(self):
        super(Sampler3D,self).__init__()

    def forward(self,input,param):
        """
        Returns the data read from the given input with the given parametrization

        Parameters
        ----------
        input : Tensor
            the input (C,W,H,D) tensor
        param : Tensor
            the given (UVW,3,) parametrization tensor

        Returns
        -------
        Tensor
            the sampled (UVW,C,) tensor
        """

        return fetch_texture3D(input,param,mode='bilinear')
