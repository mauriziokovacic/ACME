from datetime import timedelta
from .TextPlot import *


class TrainStatPlot(TextPlot):
    """
    A class representing a textual representation of the training stats
    """

    def __init__(self, session, win='TrainStatPlot'):
        """
        Parameters
        ----------
        session : Visdom
            the visdom session
        win : str (optional)
            the window id (default is 'TrainStatPlot')
        """

        super(TrainStatPlot, self).__init__(session, win=win)

    def update(self, epoch, train, iteration, t):
        """
        Updates the figure content

        Parameters
        ----------
        epoch : tuple
            the (current, end) epoch tuple
        train : tuple
            the (current, end) training set tuple
        iteration : tuple
            the (current, end) iteration tuple
        t : float
            the iteration processing time

        Returns
        -------
        None
        """

        e = epoch
        j = train
        i = iteration
        g = (e[0] * j[0] * i[1] + i[0], e[1] * j[1] * i[1])
        text = ['Iteration    :\t {}/{} \t({:.2f}%)'.format(i[0]+1, i[1], ((i[0] + 1) / i[1]) * 100),
                'Train        :\t {}/{} \t({:.2f}%)'.format(j[0]+1, j[1], ((j[0] + 1) / j[1]) * 100),
                'Epoch        :\t {}/{} \t({:.2f}%)'.format(e[0]+1, e[1], ((e[0] + 1) / e[1]) * 100),
                'Total        :\t {}/{} \t({:.2f}%)'.format(g[0]+1, g[1], ((g[0] + 1) / g[1]) * 100),
                'Elapsed      :\t {}'.format(timedelta(seconds=t * g[0])),
                'Iteration ETA:\t {}'.format(timedelta(seconds=t)),
                'Train     ETA:\t {}'.format(timedelta(seconds=t * (i[1] - i[0]))),
                'Epoch     ETA:\t {}'.format(timedelta(seconds=t * ((j[1] - j[0]) * i[1]) * (i[1] - i[0]))),
                'Total     ETA:\t {}'.format(timedelta(seconds=t * (g[1] - g[0])))]
        super(TrainStatPlot, self).update(text)
