class Transform(object):
    """
    A proper abstract interface for the torch_geometric transforms

    The Transform is an inplace operation performed over the Data objects contained in a dataset

    Methods
    -------
    eval(x, *args, **kwargs)
        evaluates the transform on the given data
    __eval__(x, *args, **kwargs)
        evaluates the transform on the given data
    __extra_repr__()
        adds extra info to the standard repr
    """

    def __init__(self):
        pass

    def eval(self, x, *args, **kwargs):
        """
        Evaluates the transform on the given data (public interface)

        Parameters
        ----------
        x : Data
            the input data
        args : ...
            optional positional arguments
        kwargs : ...
            optional keyword arguments

        Returns
        -------
        Data
            the transformed data

        Raises
        ------
        NotImplementedError
            if object does not implement __eval__ method
        """

        self.__eval__(x, *args, **kwargs)
        return x

    def __eval__(self, x, *args, **kwargs):
        """
        Evaluates the transform on the given data

        Parameters
        ----------
        x : Data
            the input data
        args : ...
            optional positional arguments
        kwargs : ...
            optional keyword arguments

        Returns
        -------
        Data
            the transformed data

        Raises
        ------
        NotImplementedError
            if object does not implement this method
        """

        raise NotImplementedError

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.__extra_repr__())

    def __extra_repr__(self):
        """
        Adds extra info to the standard repr

        By default the extra repr is an empty string

        Returns
        -------
        str
            the extra info
        """
        return ''

    def __call__(self, *args, **kwargs):
        return self.eval(*args, **kwargs)
