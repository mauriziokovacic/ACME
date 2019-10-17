import numpy
from ..VisdomFigure import *


class LineTracerPlot(PlotlyFigure):
    """
    A class representing a line tracer plot

    Attributes
    ----------
    __line : dict
        the contained lines dictionary
    limit : int
        the maximum number of samples to plot
    """

    def __init__(self, session, win='LineTracerPlot', title=None, limit=None, x_label='X', y_label='Y', **kwargs):
        """
        Parameters
        ----------
        session : Visdom
            the visdom session
        win : str (optional)
            the window id (default is 'LineTracerPlot')
        title : str (optional)
            the title of the plot (default is 'LineTracerPlot')
        limit : int (optional)
            the maximum number of samples to plot (default is None)
        x_label : str (optional)
            the x axis label (default is 'X')
        y_label : str (optional)
            the y axis label (default is 'Y')
        kwargs : ...
        """

        super(LineTracerPlot, self).__init__(session, win=win)
        self.limit = limit if limit is None else -limit
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
            the x component of a sample
        y : float
            the y component of a sample
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
            self.__fig__.add_scatter(x=numpy.array([x]), y=numpy.array([y]), mode='lines', name=legend)
        else:
            i = self.__line[name]
            self.__fig__.data[i].x = numpy.append(self.__fig__.data[i].x[self.limit:], x)
            self.__fig__.data[i].y = numpy.append(self.__fig__.data[i].y[self.limit:], y)

    def __getattr__(self, key):
        value = self.__dict__[key]
        if key == 'limit':
            value = -value
        return value

    def __setattr__(self, key, value):
        if key == 'limit':
            if value is not None:
                value = -value
        self.__dict__[key] = value
