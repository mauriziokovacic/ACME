from ...math.Range import *


class Metric(object):
    def __init__(self, name='Metric',
                 dimension='1',
                 acceptable_range=Range(),
                 normal_range=Range(),
                 full_range=Range(),
                 q_for_unit=None):
        self.name             = name
        self.dimension        = dimension
        self.acceptable_range = acceptable_range
        self.normal_range     = normal_range
        self.full_range       = full_range
        self.q_for_unit       = q_for_unit

    def eval(self, *args, **kwargs):
        raise NotImplementedError

    def is_in_acceptable_range(self, value):
        return self.acceptable_range(value)

    def is_in_normal_range(self, value):
        return self.normal_range(value)

    def is_in_full_range(self, value):
        return self.full_range(value)

    def __contains__(self, value):
        return self.is_in_acceptable_range(value=value), \
               self.is_in_normal_range(value=value), \
               self.is_in_full_range(value=value)

    def __call__(self, *args, **kwargs):
        return self.eval(*args, **kwargs)

    def __repr__(self):
        text = []
        text += ['║ ' + self.name.upper()]
        text += ['║ Dimension          : ' + str(self.dimension)]
        text += ['║ Acceptable Range   : ' + str(self.acceptable_range)]
        text += ['║ Normal Range       : ' + str(self.normal_range)]
        text += ['║ Full Range         : ' + str(self.full_range)]
        text += ['║ q for unit simplex : ' + str(self.q_for_unit)]
        n = max([len(t) for t in text])
        text = [t + ' ' * (n - len(t)) for t in text]
        text = [t + ' ║' for t in text]
        text.insert( 0, '╔' + '═' * n + '╗')
        text.insert( 2, '╠' + '═' * n + '╣')
        text.append('╚' + '═' * n + '╝')
        return '\n'.join(text)
