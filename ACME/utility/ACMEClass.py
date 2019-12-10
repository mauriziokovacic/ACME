from .hasmethod import *


class ACMEClass(object):
    """
    The base class representation for all the ACME lib classes
    """

    def __init__(self, *args, **kwargs):
        pass

    def __repr__(self):
        """
        Returns the string representation of the class.

        If the derived class has an '__extra_repr__(self)' method, its returned string will be added within the parenthesis

        Returns
        -------
        str
            the string representation of the class
        """

        return '{}({})'.format(self.__class__.__name__, self.__extra_repr__() if hasmethod(self, '__extra_repr__') else '')

    def cast(self, dtype, *args, **kwargs):
        """
        Converts the object to the given type. If the conversion is not defined then an error is raised.

        Parameters
        ----------
        dtype : type
            the type to convert the object to

        Returns
        -------
        object
            the converted object

        Raises
        ------
        RuntimeError
            if the conversion is not defined
        """

        method = '__{}__'.format(dtype if isinstance(dtype, str) else dtype.__class__.__name__)
        if hasmethod(self, method):
            return eval('self.{}(*args, **kwargs)'.format(method))
        else:
            raise RuntimeError('Unknown type conversion')
