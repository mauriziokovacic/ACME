import numpy
from .LineTracerPlot import *


class LossPlot(LineTracerPlot):
    """
    A class representing a loss plot
    """
    
    def __init__(self, session, win='LossPlot', limit=None):
        """
        Parameters
        ----------
        session : Visdom
            the visdom session
        win : str
            the window id
        limit : int (optional)
            the maximum number of samples to plot (default is None)
        """

        super(LossPlot, self).__init__(session,
                                       win=win,
                                       limit=limit,
                                       title='Global Loss',
                                       x_label='Iteration', y_label='Loss')

    def __update__(self, loss_dict):
        """
        Updates the plot

        Parameters
        ----------
        loss_dict : dict
            the loss dictionary

        Returns
        -------
        LossPlot
            the plot itself
        """

        x = 0
        if self.__fig__.data:
            x = len(self.__fig__.data[0].x)
        for n, v in loss_dict.items():
            super(LossPlot, self).__update__(n, x, v)

