from ...utility.islist  import *
from ...utility.istuple import *
from ..VisdomFigure     import *


class TextPlot(VisdomFigure):
    """
    A class representing a text plot
    """

    def __init__(self, session, win='TextPlot'):
        """
        Parameters
        ----------
        session : Visdom
            the visdom session
        win : str (optional)
            the window id (default is 'TextPlot')
        """
        super(TextPlot, self).__init__(session, win=win)

    def update(self, text, **kwargs):
        """
        Updates the figure content

        Parameters
        ----------
        text : str or list or tuple
            the text to be plotted
        kwargs : ...

        Returns
        -------
        None
        """

        if islist(text) or istuple(text):
            h = self.session.text('', win=self.win, **kwargs)
            for t in text:
                h = self.session.text(t, win=h, append=True)
            return h
        return self.session.text(text, win=self.win, **kwargs)
