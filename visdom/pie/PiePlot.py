from ..VisdomFigure import *


class PiePlot(PlotlyFigure):
    """
    A class representing a pie plot
    """

    def __init__(self, session, win='PiePlot', title=None):
        """
        Parameters
        ----------
        session : Visdom
            the visdom session
        win : str (optional)
            the window id (default is 'PiePlot')
        title : str (optional)
            the plot title (default is 'PiePlot')
        """

        super(PiePlot, self).__init__(session, win=win)
        self.__fig__.update_layout(title={'text': title if title is not None else win})

    def __update__(self, labels, values, **kwargs):
        """
        Updates the figure content

        Parameters
        ----------
        labels : list
            a list of strings containing the values labels
        values : list
            a list of values to plot
        kwargs : ...
            the keyword arguments of the pie plot

        Returns
        -------
        None
        """

        if self.__fig__.data:
            self.__fig__.data = []
        self.__fig__.add_pie(labels=labels, values=values, sort=False, **kwargs)
