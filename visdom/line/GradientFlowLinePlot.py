from ...model.grad_flow import *
from .LinePlot          import *


class GradientFlowLinePlot(LinePlot):
    """
    A class representing the gradient flow plot for a given model
    """

    def __init__(self, session, win='GradientFlowLinePlot'):
        """
        Parameters
        ----------
        session : Visdom
            the visdom session
        win : str (optional)
            the window id (default is 'GradientFlowLinePlot')
        """

        super(GradientFlowLinePlot, self).__init__(
            session, win=win,
            title='Gradient Flow',
            x_label='Net Depth', y_label='Grad Norm')

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
        x = [i for i in range(len(g))]
        y = [v for v in g.values()]
        super(GradientFlowLinePlot, self).__update__(name='grad', x=x, y=y, legend='Grad')
        return