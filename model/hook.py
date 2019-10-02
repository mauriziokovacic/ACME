class Hook(object):
    """
    A class representing a hook for PyTorch modules

    Attributes
    ----------
    __handle : object
        the hook handle
    layerFcn : callable
        a callable function to run over the hooked layer
    inputFcn : callable
        a callable function to run over the hooked layer input
    outputFcn : callable
        a callable function to run over the hooked layer output
    name : str
        a name for the hook

    Methods
    -------
    is_bound()
        returns True if the hood has been bound to a layer, False otherwise
    bind(layer)
        binds the hook to a given layer, unbinding it from the previous if needed
    unbind()
        unbinds the hook from the hooked layer
    eval(layer, input, output)
        evaluates layerFcn, inputFcn and outputFcn over the bound layer
    """

    def __init__(self, layer=None, layerFcn=None, inputFcn=None, outputFcn=None, name='Hook'):
        """
        Parameters
        ----------
        layer : torch.nn.Module (optional)
            a layer to hook (default is None)
        layerFcn : callable (optional)
            a callable function to run over the hooked layer
        inputFcn : callable (optional)
            a callable function to run over the hooked layer input
        outputFcn : callable (optional)
            a callable function to run over the hooked layer output
        name : str (optional)
            a name for the hook
        """

        self.__handle  = None
        self.bind(layer)
        self.layerFcn  = layerFcn
        self.inputFcn  = inputFcn
        self.outputFcn = outputFcn
        self.name      = name

    def __del__(self):
        self.unbind()

    def is_bound(self):
        """
        Returns True if the hood has been bound to a layer, False otherwise

        Returns
        -------
        bool
            True if the hook has been bound, False otherwise
        """

        return self.__handle is not None

    def bind(self, layer):
        """
        Binds a hook to a given layer, unbinding it from the previous if needed

        Parameters
        ----------
        layer : torch.nn.Module
            a layer to hook

        Returns
        -------
        Hook
            the hook itself
        """

        if layer is not None:
            if self.is_bound():
                self.unbind()
            self.__handle = layer.register_forward_hook(self.eval)
        return self

    def unbind(self):
        """
        Unbinds the hook from the hooked layer

        Returns
        -------
        Hook
            the hook itself
        """

        if self.__handle is not None:
            self.__handle.remove()
        self.__handle = None
        return self

    def eval(self, layer, input, output):
        """
        Evaluates layerFcn, inputFcn and outputFcn over the bound layer

        Parameters
        ----------
        layer : torch.nn.Module
            the hooked layer
        input : object
            the layer input
        output : object
            the layer output

        Returns
        -------
        None
        """

        if self.layerFcn is not None:
            self.layerFcn(layer)
        if self.inputFcn is not None:
            self.inputFcn(input)
        if self.outputFcn is not None:
            self.outputFcn(output)

    def __repr__(self):
        """
        Returns the hook name

        Returns
        -------
        str
            the hook name
        """

        return self.name


class DeferredHook(Hook):
    """
    A class representing a deferred hook. It will store both the input and the output of the hooked layer to be
    processed at a different time. The standard behaviours of the base Hook are preserved

    Attributes
    ----------
     __handle : object
        the hook handle
    layerFcn : callable
        a callable function to run over the hooked layer
    inputFcn : callable
        a callable function to run over the hooked layer input
    outputFcn : callable
        a callable function to run over the hooked layer output
    name : str
        a name for the hook
    input : object
        the input of the hooked layer (default is None)
    output : object
        the output of the hooked layer (default is None)
    """

    def __init__(self, *args, **kwargs):
        super(DeferredHook, self).__init__(*args, **kwargs)
        self.input  = None
        self.output = None

    def eval(self, layer, input, output):
        """
        Evaluates layerFcn, inputFcn and outputFcn over the bound layer, storing its input and output afterwards

        Parameters
        ----------
        layer : torch.nn.Module
            the hooked layer
        input : object
            the layer input
        output : object
            the layer output

        Returns
        -------
        None
        """

        super(DeferredHook, self).eval(layer, input, output)
        self.input  =  input
        self.output = output

