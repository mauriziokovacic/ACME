import plotly.graph_objs as go


class VisdomFigure(object):
    """
    A class representing a Visdom generic figure

    Attributes
    ----------
    session : Visdom
        the visdom session the figure belongs to
    __win__ : str
        the window id

    Methods
    -------
    close()
        closes the figure
    """

    def __init__(self, session, win='Figure'):
        self.session = session
        self.__win__ = win

    def update(self, *args, **kwargs):
        return self

    def close(self):
        """
        Closes the figure

        Returns
        -------
        None
        """

        self.session.close(self.__win__)


class PlotlyFigure(VisdomFigure):
    """
    A class representing a plotly figure adapted for visdom

    Attributes
    ----------
    __fig__ : plotly.Figure
        the plotly figure

    Methods
    -------
    update(*args, **kwargs)
        updates the figure content
    __update__(*args, **kwargs)
        updates the figure content
    """

    def __init__(self, session, win='Figure'):
        """
        Parameters
        ----------
        session : Visdom
            the visdom session
        win : str (optional)
            the window id (default is 'Figure')
        """

        super(PlotlyFigure, self).__init__(session, win=win)
        self.__fig__ = go.Figure()

    def update(self, *args, **kwargs):
        """
        Updates the figure content

        Parameters
        ----------
        args
        kwargs

        Returns
        -------
        PlotlyFigure
            the figure itself
        """
        self.__update__(*args, **kwargs)
        self.session.plotlyplot(self.__fig__, win=self.__win__)
        return self

    def __update__(self, *args, **kwargs):
        raise NotImplementedError('Derived class should implement the __update__ method')
