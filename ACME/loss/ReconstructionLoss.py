from ..utility.strcmpi import *
from .Loss import *


class ReconstructionLoss(Loss):
    """
    A class representing the standard autoencoder reconstruction loss

    Attributes
    ----------
    fcn : str
        the reconstruction function to apply, either 'bce' or 'mse'
    """

    def __init__(self, *args, fcn='bce', name='Reconstruction', **kwargs):
        """
        Parameters
        ----------
        fcn : str (optional)
            the reconstruction function to apply, either 'bce' for binary cross entropy or
            'mse' for mean squared error (default is 'bce')
        """

        super(ReconstructionLoss, self).__init__(*args, name=name, **kwargs)
        self.__evalFcn = []
        self.__fcn     = fcn

    def __eval__(self, x, x_hat):
        return self.__evalFcn(x_hat, x)

    @property
    def fcn(self):
        return self.__fcn

    @fcn.setter
    def fcn(self, value):
        if strcmpi(value, 'bce'):
            self.__fcn     = value
            self.__evalFcn = torch.nn.BCELoss()
            return
        if strcmpi(value, 'mse'):
            self.__fcn     = value
            self.__evalFcn = torch.nn.MSELoss()
            return
        raise RuntimeError('Unknown value for type. Expected ''bce'' or ''mse'', but got {}.'.format(value))
