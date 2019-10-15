from ..utility.islist  import *
from ..utility.istuple import *


def plot_text(session, text, win='Text', **kwargs):
    """
    Plots the given text on visdom

    Parameters
    ----------
    session : Visdom
        a visdom session
    text : str or list or tuple
        the text or text lines to plot
    win : str (optional)
        the visdom window name (default is 'Text')
    kwargs : ...

    Returns
    -------
    object
        the plot object
    """

    if islist(text) or istuple(text):
        h = session.text('', win=win, **kwargs)
        for t in text:
            h = session.text(t, win=h, append=True)
        return h
    return session.text(text, win=win, **kwargs)
