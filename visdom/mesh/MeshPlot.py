import plotly.figure_factory as FF
from ...utility.torch2numpy import *
from ..VisdomFigure         import *


class MeshPlot(PlotlyFigure):
    """
    A class representing a 3D mesh plot

    Attributes
    ----------
    __title : str
        the plot title
    """

    def __init__(self, session, win='MeshPlot', title=None):
        """
        Parameters
        ----------
        session : Visdom
            the visdom session
        win : str (optional)
            the window id (default is 'MeshPlot')
        title : str (optional)
            the plot title
        """

        super(MeshPlot, self).__init__(session, win=win)
        self.__title = title if title is not None else win

    def __update__(self, P, T, color='rgb(255,0,0)'):
        """
        Updates the figure content

        Parameters
        ----------
        P : Tensor
            the (N,3,) points set tensor
        T : LongTensor
            the (3,M,) topology tensor
        color : str (optional)
            the mesh color

        Returns
        -------
        None
        """
        p = torch2numpy(P)
        t = torch2numpy(T.t())
        fig = FF.create_trisurf(x=p[:, 0], y=p[:, 1], z=p[:, 2], simplices=t,
                                show_colorbar=False, colormap=color)
        fig.data[0].flatshading = True
        fig.layout = {'title': {'text': self.__title}}
