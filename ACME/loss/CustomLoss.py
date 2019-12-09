from .Loss import *


class CustomLoss(Loss):
    """
    A class representing a custom loss, evaluating a given function

    Parameters
    ----------
    fcn : callable
        a function to evaluate
    """

    def __init__(self, fcn, *args, name='Custom', **kwargs):
        """
        Parameters
        ----------
        fcn : callable
            a function to evaluate
        """

        super(CustomLoss, self).__init__(*args, name=name, **kwargs)
        self.fcn = fcn

    def __eval__(self, input, output):
        return self.fcn(input, output)
