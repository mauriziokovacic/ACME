from ..utility.ACMEClass import *
from ..utility.hasmethod import *


class Transform(ACMEClass):
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
        super(Transform, self).__init__()

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

    def __call__(self, *args, **kwargs):
        return self.eval(*args, **kwargs)
