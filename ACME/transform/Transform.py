from ..utility.hasmethod import *


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
        """
        Returns the string representation of the Transform class.

        If the derived class has an 'extra_repr(self)' method, its returned string will be added within the parenthesis

        Returns
        -------
        str
            the string representation of the class
        """

        return '{}({})'.format(self.__class__.__name__, self.extra_repr() if hasmethod(self, 'extra_repr') else '')

    def __call__(self, *args, **kwargs):
        return self.eval(*args, **kwargs)
