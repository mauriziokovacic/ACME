import numpy
from ..VisdomFigure import *


class LinePlot(PlotlyFigure):
    """
    A class representing a line plot

    Attributes
    ----------
    __line : dict
        the contained lines dictionary
    """

    def __init__(self, session, win='Line', title=None, x_label='X', y_label='Y', **kwargs):
        """
        Parameters
        ----------
        session : Visdom
            the visdom session
        win : str (optional)
            the window id (default is 'Line')
        title : str (optional)
            the title of the plot (default is 'Line')
        x_label : str (optional)
            the x axis label (default is 'X')
        y_label : str (optional)
            the y axis label (default is 'Y')
        kwargs : ...
        """

        super(LinePlot, self).__init__(session, win=win)
        self.__line = {}
        self.__fig__.update_layout(title={'text': title if title is not None else win},
                                   xaxis_title=x_label,
                                   yaxis_title=y_label)

    def __update__(self, name, x, y, legend=None):
        """
        Updates the figure content

        Parameters
        ----------
        name : str
            the line name
        x : float
            the line samples x components
        y : float
            the line samples y components
        legend : str (optional)
            the legend name for the line

        Returns
        -------
        None
        """

        if legend is None:
            legend = name
        if name not in self.__line:
            self.__line[name] = len(self.__line)
            self.__fig__.add_scatter(x=numpy.array(x), y=numpy.array(y), mode='lines', name=legend)
        else:
            i = self.__line[name]
            self.__fig__.data[i].x = numpy.array(x)
            self.__fig__.data[i].y = numpy.array(y)
