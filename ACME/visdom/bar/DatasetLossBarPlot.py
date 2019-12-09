import torch
from ..VisdomFigure     import *


class DatasetLossBarPlot(PlotlyFigure):
    """
    A class representing a gradient flow plot
    """

    def __init__(self, session, win='DatasetLossBarPlot'):
        """
        Parameters
        ----------
        session : Visdom
            the visdom session
        win : str (optional)
            the window id (default is 'DatasetLossBarPlot')
        """

        super(DatasetLossBarPlot, self).__init__(session, win=win)
        self.__fig__.update_layout(title={'text': 'Dataset Loss'},
                                   xaxis_title='Object',
                                   yaxis_title='Loss')

    def __update__(self, losses):
        """
        Updates the figure content

        Parameters
        ----------
        losses : list
            a list of loss values

        Returns
        -------
        None
        """

        if self.__fig__.data:
            self.__fig__.data = []
        self.__fig__.add_bar(name='Loss', x=list(range(len(losses))), y=losses)
