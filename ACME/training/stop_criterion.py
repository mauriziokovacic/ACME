import warnings
from ..utility.ACMEClass import *
from ..utility.hasmethod import *
from ..utility.isnan     import *
from ..model.is_frozen   import *


class StopCriterion(ACMEClass):
    """
    A class representing a stop criterion for the training

    Attributes
    ----------
    attr : str
        the name of the attribute to check

    Methods
    -------
    eval(**kwargs)
        evaluates the stop criterion
    """

    def __init__(self, attr):
        """
        Parameters
        ----------
        attr : str
            the name of the attribute to check
        """

        super(StopCriterion, self).__init__()
        if not hasmethod(self, '__eval__'):
            raise NotImplementedError('The derived class should implement the method:\n'
                                      ' __eval__(self, x: object) -> bool ')
        self.attr = attr

    def eval(self, **kwargs):
        """
        Evaluates the stop criterion

        Parameters
        ----------
        kwargs : ...
            a series of keyword arguments such as 'model', 'loss', 'grad'

        Returns
        -------
        bool
            True if the stop criterion as met, False otherwise

        Warnings
        --------
        RuntimeWarning
            if stop criterion is met
        """

        if self.attr in kwargs:
            if self.__eval__(kwargs[self.attr]):
                warnings.warn('{} has been evaluated positively'.format(str(self)), RuntimeWarning)
                return True
        return False

    def __repr__(self):
        """
        Returns the string representation of the StopCriterion class.

        If the derived class has an 'extra_repr(self)' method, its returned string will be added within the parenthesis

        Returns
        -------
        str
            the string representation of the class
        """

        return '{}(attr={}{})'.format(self.__class__.__name__,
                                      self.attr,
                                      self.__extra_repr__() if hasmethod(self, '__extra_repr__') else '')

    def __call__(self, **kwargs):
        return self.eval(**kwargs)


class VanishingGradientCriterion(StopCriterion):
    """
    A class representing a vanishing gradient stop criterion

    It evaluates True whenever the model has reached an almost stable gradient

    Attributes
    ----------
    tol : float
        the stop criterion tolerance
    """

    def __init__(self, tol=10e-5):
        """
        Parameters
        ----------
        tol : float (optional)
            the stop criterion tolerance (default is 10e-5)
        """

        super(VanishingGradientCriterion, self).__init__(attr='grad')
        self.tol = tol

    def __eval__(self, x):
        """
        Evaluates the stop criterion

        Parameters
        ----------
        x : float or Tensor
            the mean gradient of the model

        Returns
        -------
        bool
            True if the gradient is below the tolerance, False otherwise
        """
        return x < self.tol

    def __extra_repr__(self):
        return ', tol={}'.format(self.tol)


class FrozenModelCriterion(StopCriterion):
    """
    A class representing a frozen model stop criterion

    It evaluates True whenever the model is frozen
    """

    def __init__(self):
        super(FrozenModelCriterion, self).__init__(attr='model')

    def __eval__(self, x):
        """
        Evaluates the stop criterion

        Parameters
        ----------
        x : torch.nn.Module
            the parameter value to check

        Returns
        -------
        bool
            True if the model is frozen, False otherwise
        """

        return is_frozen(x)


class NaNLossCriterion(StopCriterion):
    """
    A class representing a NaN loss stop criterion

    It evaluates True whenever the model loss is NaN
    """

    def __init__(self):
        super(NaNLossCriterion, self).__init__(attr='loss')

    def __eval__(self, x):
        """
        Evaluates the stop criterion

        Parameters
        ----------
        x : object
            the parameter value to check

        Returns
        -------
        bool
            True if the object is NaN, False otherwise
        """
        return isnan(x)


class NoneCriterion(StopCriterion):
    """
    A criterion that checks if a specific attribute is None
    """

    def __init__(self, attr):
        """
        Parameters
        ----------
        attr : str
            the name of the attribute to check
        """

        super(NoneCriterion, self).__init__(attr=attr)

    def __eval__(self, x):
        """
        Evaluates the stop criterion

        Parameters
        ----------
        x : object
            the parameter value to check

        Returns
        -------
        bool
            True if the object is None, False otherwise
        """

        return x is None
