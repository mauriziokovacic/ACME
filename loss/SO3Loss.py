from .Loss       import *
from .so3_metric import *


class SO3Loss(Loss):
    """
    A class defining the SO(3) metric loss
    """

    def __init__(self, *args, **kwargs):
        super(SO3Loss, self).__init__(*args, name='SO3', **kwargs)

    def __eval__(self, input, output):
        return so3_metric(output)
