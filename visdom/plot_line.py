import numpy
from ..utility.static_vars import *


@static_vars(plot=None)
def plot_multiline(session,
                   line_name,
                   legend_name,
                   x, y,
                   win='MultiLine',
                   xlabel='x',
                   ylabel='y',
                   reset=False,
                   **kwargs):
    """
    Plots one or more lines on a visdom session

    Parameters
    ----------
    session : Visdom
        a visdom session
    line_name : str
        the name of the line
    legend_name : str
        the line name in the legend
    x : int or float
        the line x coordinate
    y : int or float
        the line y coordinate
    win : str (optional)
        the window name (default is 'MultiLine')
    xlabel : str (optional)
        the x axis label (default is 'x')
    ylabel : str (optional)
        the y axis label (default is 'y')
    reset : bool (optional)
        if True resets the plot (default is False)
    kwargs : ...

    Returns
    -------
    object
        the plot object
    """

    if (plot_multiline.plot is None) or reset:
        plot_multiline.plot = {}

    if line_name not in plot_multiline.plot:
        plot_multiline.plot[line_name] = session.line(X=numpy.array([x, x]),
                                                      Y=numpy.array([y, y]),
                                                      opts=dict(
                                                          legend=[legend_name],
                                                          title=win,
                                                          xlabel=xlabel,
                                                          ylabel=ylabel,)
                                                      )
    else:
        session.line(X=numpy.array([x]),
                     Y=numpy.array([y]),
                     win=plot_multiline.plots[line_name],
                     name=legend_name,
                     update='append')
