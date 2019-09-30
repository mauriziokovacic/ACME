from ...math.constant import *


class Range(object):
    def __init__(self, min=-Inf, max=Inf):
        self.min = float(min)
        self.max = float(max)

    def eval(self, value):
        return (value >= self.min) * (value <= self.max)

    def __call__(self, value):
        return self.eval(value)

    def __repr__(self):
        return '['+str(self.min)+', '+str(self.max)+']'
