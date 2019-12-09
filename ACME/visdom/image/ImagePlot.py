from ...utility.isnumpy import *
from ..VisdomFigure     import *


class ImagePlot(VisdomFigure):
    """
    A class representing an images plot
    """

    def __init__(self, session, win='ImagePlot'):
        """
        Parameters
        ----------
        session : Visdom
            the visdom session
        win : str (optional)
            the window id (default is 'ImagePlot')
        """

        super(ImagePlot, self).__init__(session, win=win)

    def update(self, images, **kwargs):
        """
        Updates the figure content

        Parameters
        ----------
        images : Tensor
            the images to plot

        Returns
        -------
        None
        """

        self.session.images(images if isnumpy(images) else images.numpy(), win=self.__win__, **kwargs)
