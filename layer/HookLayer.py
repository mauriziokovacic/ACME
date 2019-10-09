import torch
from ..utility.islist    import *
from ..layer.Concatenate import *
from ..model.hook        import *


class HookLayer(torch.nn.Module):
    """
    A class representing a layer capable of hooking itself to another layer

    Attributes
    ----------
    layer : torch.nn.Module
        the layer to evaluate
    __hook : DeferredHook
        a hook to another layer

    Methods
    -------
    is_bound()
        returns True if the layer is bound to another layer, False otherwise
    bind(layer)
        binds the layer to the given one
    unbind()
        unbinds the layer from any hooked one
    forward(*args, **kwargs)
        returns the output of the HookLayer
    """

    def __init__(self, layer, hook_layer=None, name='HookLayer'):
        """
        Parameters
        ----------
        layer : torch.nn.Module
            the layer to evaluate
        hook_layer : torch.nn.Module or list (optional)
            the layer(s) to hook (default is None)
        """

        super(HookLayer, self).__init__()
        self.add_module('layer', layer)
        self.__hook = {}
        self.name   = name
        self.bind(hook_layer)

    def is_bound(self):
        """
        Returns True if the layer is bound to another layer, False otherwise

        Returns
        -------
        bool
            True if the layer is bound to another layer, False otherwise
        """

        return all([h.is_bound() for h in self.__hook.values()])

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

        hook = layer
        if not islist(hook):
            hook = [hook]
        for h in hook:
            name = 'hook_{}'.format(len(self.__hook))
            self.__hook[name] = DeferredHook(layer=h, name=name)
        return self

    def unbind(self, i=None):
        """
        Unbinds the layer from the specified hooked layer

        Parameters
        ----------
        i : int (optional)
            the index of the hook to remove. If None all the layers will be unbound (default is None)

        Returns
        -------
        HookLayer
            the layer itself
        """

        if i is None:
            i = list(range(len(self.__hook)))
        for ii in i:
            key = 'hook_{}'.format(ii)
            self.__hook[key].unbind()
            del self.__hook[key]
        return self

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

        return self.layer(*args, *tuple(h.output for h in self.__hook.values()), **kwargs)


class ResidualLayer(HookLayer):
    """
    A class representing a residual layer.
    """

    def __init__(self, layer, operation='cat', dim=1, **kwargs):
        """
        Parameters
        ----------
        layer : torch.nn.Module
            the layer to evaluate
        operation : str (optional)
            the operation to perform. It must be one of the following:
            'cat', 'add', 'mean', 'min', 'max' or 'std' (default is 'cat')
        dim : int (optional)
            the dimension along the residual operation is performed (default is 0)
        kwargs
        """

        fun = {
            'cat':  lambda x: torch.nn.Sequential(Concatenate(dim=dim), x),
            'add':  lambda x: torch.nn.Sequential(AddLayer(dim=dim), x),
            'mean': lambda x: torch.nn.Sequential(MeanLayer(dim=dim), x),
            'min':  lambda x: torch.nn.Sequential(MinLayer(dim=dim), x),
            'max':  lambda x: torch.nn.Sequential(MaxLayer(dim=dim), x),
            'std':  lambda x: torch.nn.Sequential(StdLayer(dim=dim), x),
        }

        if operation.lower() in fun:
            l = fun[operation.lower()](layer)
        else:
            raise ValueError('Input operation must be one of the following:\n{}'.format('\n'.join(['cat', 'add', 'mean', 'min', 'max' or 'std'])))
        super(ResidualLayer, self).__init__(l, **kwargs)

