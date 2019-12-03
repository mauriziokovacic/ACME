from .Loss             import *
from .hausdorff_metric import *


class HausdorffLoss(Loss):
    """
    A class defining the Hausdorff metric loss
    """

    def __init__(self, *args, name='Hausdorff', **kwargs):
        super(HausdorffLoss, self).__init__(*args, name=name, **kwargs)

    def __eval__(self, input, output):
        return hausdorff_metric(input, output)