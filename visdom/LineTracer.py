import numpy


class LineTracer(object):
    def __init__(self, session, win='LineTracer', title=None, x_label='X', y_label='Y'):
        self.session = session
        self.__win   = win
        self.title   = title if title is not None else win
        self.x_label = x_label
        self.y_label = y_label
        self.__line = {}

    def add(self, name, x, y, legend=None):
        if name in self.__line:
            return self.update(name, x, y)
        if legend is None:
            legend = name
        self.__line[name] = legend
        self.session.line(
            X=[x, x],
            Y=[y, y],
            win=self.__win,
            opts=dict(
                legend=[legend],
                title=self.title,
                xlabel=self.x_label,
                ylabel=self.y_label
            )
        )
        return self

    def update(self, name, x, y):
        if name not in self.__line:
            return self.add(name, x, y)
        self.session.line(
            X=numpy.array([x]),
            Y=numpy.array([y]),
            win=self.__win,
            name=self.__line[name],
            update='append',
        )
        return self
