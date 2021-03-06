from datetime import timedelta
from .PiePlot import *


class TrainIterPiePlot(PiePlot):
    def __init__(self, session, win='TrainIterPiePlot'):
        super(TrainIterPiePlot, self).__init__(session, win=win, title='Training Process')

    def __update__(self, epoch, iter, time=None, **kwargs):
        d = epoch[0] / epoch[1]
        e = 1 / epoch[1]
        t = 1 - (e+d)
        i = e * ((iter[0] + 1) / iter[1])
        if i == e:
            i = 0
            d = d + e
        e = 1 - (t+d+i)
        if time is not None:
            g = (epoch[0] * iter[1] + iter[0], epoch[1] * iter[1])
            self.__fig__.update_layout(title={'text': 'ETA: {}'.format(timedelta(seconds=time * (g[1]-g[0])))})
        super(TrainIterPiePlot, self).__update__(
            labels=['Done', 'Processing', 'Epoch', 'Missing'],
            values=[d, i, e, t],
            marker_colors=['rgb(34, 177, 76)',   # green
                           'rgb(255, 163, 23)',  # orange
                           'rgb(237, 28, 36)',   # red
                           'rgb(0, 102, 232)'],  # blue
            **kwargs
        )
