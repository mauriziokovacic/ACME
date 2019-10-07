import torch
from ..model.hook import *


class HookLayer(torch.nn.Module):
    """
    A class representing a layer capable of hooking itself to another layer

    Attributes
    ----------
    __hook : Hook
        a hook to another layer

    Methods
    -------
    is_bound()
        returns True if the layer is bound to another layer, False otherwise
    bind(layer)
        binds the layer to the given one
    unbind()
        unbinds the layer from any hooked one
    __outputFcn(output)
        stores the output of the bound layer
    forward(*args, z, **kwargs)
        returns the output of this layer
    """

    def __init__(self, layer=None):
        """
        Parameters
        ----------
        layer : torch.nn.Module (optional)
            the layer to bind
        """

        super(HookLayer, self).__init__()
        self.__hook = Hook(layer=layer, outputFcn=self.__outputFcn)

    def is_bound(self):
        """
        Returns True if the layer is bound to another layer, False otherwise

        Returns
        -------
        bool
            True if the layer is bound to another layer, False otherwise
        """

        return self.__hook.is_bound()

    def bind(self, layer):
        """
        Binds the layer to the given one

        Parameters
        ----------
        layer : torch.nn.Module
            a layer to bind

        Returns
        -------
        HookLayer
            the layer itself
        """

        self.__hook.bind(layer)
        return self

    def unbind(self):
        """
        Unbinds the layer from any hooked one

        Returns
        -------
        HookLayer
            the layer itself
        """

        self.__hook.unbind()
        return self

    def __outputFcn(self, output):
        """
        Stores the output of the bound layer

        Parameters
        ----------
        output : Tensor
            the output of the bound layer

        Returns
        -------
        None
        """

        self.__output = output

    def forward(self, *args, z, **kwargs):
        """
        Returns the output of this layer

        Parameters
        ----------
        args : ...
            the inputs of this layer
        z : Tensor
            the output of the bound layer
        kwargs : ...
            the keyword inputs of this layer

        Raises
        ------
        NotImplementedError
            if the layer does not implement this method
        """

        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, self.__output, **kwargs)