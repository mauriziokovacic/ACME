import numpy
from ...model.grad_flow import *
from ..VisdomFigure     import *


class GradientFlowPlot(PlotlyFigure):
    """
    A class representing a gradient flow plot
    """

    def __init__(self, session, win='GradientFlow'):
        super(GradientFlowPlot, self).__init__(session, win=win)
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
        if self.fig.data:
            self.fig.data = []
        self.fig.add_bar(name='Min', x=list(g.keys()), y=[v[0] for v in g.values()])
        self.fig.add_bar(name='Mean', x=list(g.keys()), y=[v[1] for v in g.values()])
        self.fig.add_bar(name='Max', x=list(g.keys()), y=[v[2] for v in g.values()])
