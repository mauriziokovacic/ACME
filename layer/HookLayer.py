import torch
from ..model.hook import *


class HookLayer(torch.nn.Module):
    """
    A class representing a layer capable of hooking itself to another layer

    Attributes
    ----------
    layer : torch.nn.Module
        the layer to evaluate
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
    forward(*args, **kwargs)
        returns the output of the HookLayer
    """

    def __init__(self, layer, hook_layer=None):
        """
        Parameters
        ----------
        layer : torch.nn.Module (optional)
            the layer to bind
        """

        super(HookLayer, self).__init__()
        self.layer  = layer
        self.__hook = Hook(layer=hook_layer, outputFcn=self.__outputFcn)
        self.add_module('layer', self.layer)

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

    def forward(self, *args, **kwargs):
        """
        Returns the output of this layer

        Parameters
        ----------
        args : ...
            the inputs of this layer
        kwargs : ...
            the keyword inputs of this layer

        Returns
        -------
        Tensor
            the output of the HookLayer
        """

        return self.layer(*args, self.__output, **kwargs)
