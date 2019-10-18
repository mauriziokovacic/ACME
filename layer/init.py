import math
from ..utility.static_vars import *


def _constant(parameter, value):
    parameter.data.fill_(value)


def _zero(parameter):
    _constant(parameter, 0)


def _one(parameter):
    _constant(parameter, 1)


def _normal(parameter, mean, std):
    parameter.data.normal_(mean, std)


def _uniform(parameter, bound):
    parameter.data.uniform_(-bound, bound)


def _kaiming(parameter, fan, a):
    _uniform(parameter, math.sqrt(6 / ((1 + a ** 2) * fan)))


def _glorot(parameter):
    _uniform(parameter, math.sqrt(6.0 / (parameter.size(-2) + parameter.size(-1))))


@static_vars(fun=None)
def init(parameter, type, *args, **kwargs):
    """
    Initialize the given parameters with the specified initialization type

    Parameters
    ----------
    parameter : torch.nn.Parameter
        the parameter to initialize
    type : str
        the initialization type. Should be one between:
        'constant', 'zero', 'one', 'normal', 'kaiming', 'glorot' (or 'xavier')
    args : ...
        the arguments of the specified initialization type
    kwargs : ...
        the keyword arguments of the specified initialization type

    Returns
    -------

    """
    if not init.fun:
        init.fun = {
            'constant': _constant,
            'zero': _zero,
            'one': _one,
            'normal': _normal,
            'kaiming': _kaiming,
            'glorot': _glorot,
            'xavier': _glorot,
        }
    if parameter is not None:
        if type not in init.fun:
            raise ValueError('Type unknown. Please choose among one of the following: {}'.format('\n'.join(list(init.fun.keys()))))
        init.fun[type](parameter, *args, **kwargs)
    return parameter
