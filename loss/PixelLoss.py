from .Loss import *


class PixelLoss(Loss):
    """
    A class defining the pixel metric loss
    """

    def __init__(self, *args, **kwargs):
        super(PixelLoss, self).__init__(*args, name='Pixel', **kwargs)

    def __eval__(self, input, output):
        return torch.mean(torch.sum(torch.pow(input-output, 2), dim=(-2, -1)))
