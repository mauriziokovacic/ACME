import torch
from ...model.grad_flow import *
from ..VisdomFigure     import *


class GradientFlowBarPlot(PlotlyFigure):
    """
    A class representing a gradient flow plot
    """

    def __init__(self, session, win='GradientFlowBarPlot'):
        """
        Parameters
        ----------
        session : Visdom
            the visdom session
        win : str (optional)
            the window id (default is 'GradientFlowBarPlot')
        """

        super(GradientFlowBarPlot, self).__init__(session, win=win)
        self.__fig__.update_layout(title={'text': 'Gradient Flow'},
                                   xaxis_title='Layer',
                                   yaxis_title='Norm',
                                   yaxis_type='log')

    def __update__(self, model):
        """
        Updates the figure content

        Parameters
        ----------
        model : torch.nn.Module
            a model to plot the gradient flow from

        Returns
        -------
        None
        """

        g = grad_flow(model)
        if self.__fig__.data:
            self.__fig__.data = []
        x = list(g.keys())
        y = torch.tensor([v for v in g.values()], dtype=torch.float)
        m = y.mean()
        if m > 0:
            y[y == 0] = -m
            y = list(y)
        else:
            y = [-1, ] * y.size(0)
        self.__fig__.add_bar(name='Mean', x=x, y=y,
                             marker_color=['teal' if v > 0 else 'crimson' for v in y])
