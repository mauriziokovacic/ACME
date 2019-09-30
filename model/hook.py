import warnings


class Hook(object):
    """
    A class representing a hook for PyTorch modules

    Attributes
    ----------
    inputFcn : callable (optional)
        a callable function returning a bool if the layer input is ok (default is None)
    outputFcn : callable (optional)
        a callable function returning a bool if the layer output is ok (default is None)
    name : str (optional)
        a name for the hook
    mode : str (optional)
        either 'debug' or 'release' (default is 'debug'). In debug mode the hook will raise an error if checks on
        either the input or the output of the layer fails. In release mode a warning will be issued instead.
    value : object
        the output of the bound layer


    Methods
    -------

    """

    def __init__(self, layer=None, inputFcn=None, outputFcn=None, name='Hook', mode='debug'):
        """
        Parameters
        ----------
        layer : torch.nn.Module (optional)
            a layer to hook (default is None)
        inputFcn : callable (optional)
            a callable function returning a bool if the layer input is ok (default is None)
        outputFcn : callable (optional)
            a callable function returning a bool if the layer output is ok (default is None)
        name : str (optional)
            a name for the hook
        mode : str (optional)
            either 'debug' or 'release' (default is 'debug'). In debug mode the hook will raise an error if checks on
            either the input or the output of the layer fails. In release mode a warning will be issued instead.

        """

        self.inputFcn  = inputFcn
        self.outputFcn = outputFcn
        self.name      = name
        self.value     = None
        self.__handle  = None
        self.mode      = mode
        self.bind(layer)

    def __del__(self):
        self.unbind()

    def is_bound(self):
        return self.__handle is not None

    def bind(self, layer):
        """
        Binds the hooks to the given layer (unbinding it from previous)

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
            self.__handle = layer.register_forward_hook(self.__eval)
        return self

    def unbind(self):
        """
        Unbinds the hook from the bound layer

        Returns
        -------
        Hook
            the hook itself
        """

        if self.__handle is not None:
            self.__handle.remove()
        self.__handle = None
        return self

    def __eval(self, source, input, output):
        """


        Parameters
        ----------
        source : torch.nn.Module
            the hooked layer
        input : object
            the layer input
        output : object
            the layer output

        Raises
        ------
        RuntimError
            if the hook is in debug mode and the input/output fails to pass the checks
        """
        self.value = output
        if self.inputFcn is not None:
            if not self.inputFcn(input):
                if self.mode == 'debug':
                    raise RuntimeError('Unexpected input value for Hook: {}'.format(self.name))
                if self.mode == 'release':
                    warnings.warn('Unexpected input value for Hook: {}'.format(self.name))
        if self.outputFcn is not None:
            if not self.outputFcn(output):
                raise RuntimeError('Unexpected output value')

    def __repr__(self):
        """
        Returns the hook name

        Returns
        -------
        str
            the hook name
        """
        return self.name

    def __setattr__(self, key, value):
        if key == 'mode':
            value = value.lower()
            if value not in ['debug', 'release']:
                raise RuntimeError('mode should be either ''debug'' or ''release''')
        self.__dict__[key] = value