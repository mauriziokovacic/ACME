import numpy
from ...model.grad_flow import *
from ..VisdomFigure     import *


class GradientFlowBarPlot(PlotlyFigure):
    """
    A class representing a gradient flow plot
    """

    def __init__(self, session, win='GradientFlow'):
        super(GradientFlowBarPlot, self).__init__(session, win=win)
        self.__fig__ = go.Figure({'layout': {'title': {'text': 'Gradient Flow'}},
                                  'xaxis_title': 'Layer',
                                  'yaxis_title': 'Norm',
                                  'yaxis_type': 'log'})

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
        self.__fig__.add_bar(name='Mean', x=list(g.keys()), y=[v for v in g.values()])
