import torch


class BBoxAdapter(torch.nn.Module):
    """
    A class representing a module that adapts the bounding boxes of two input models

    Methods
    -------
    forward(x, y)
        adapts the bounding box of the y model to the x model
    """

    def __init__(self):
        super(BBoxAdapter, self).__init__()

    def forward(self, x, y):
        """
        Adapts the bounding box of the y model to the x model

        Parameters
        ----------
        x : Data
            the target input model
        y : Data
            the source input model

        Returns
        -------
        Data
            the modified source model
        """

        min   = [torch.min(x.pos, dim=0, keepdim=True)[0],
                 torch.min(y.pos, dim=0, keepdim=True)[0]]
        max   = [torch.max(x.pos, dim=0, keepdim=True)[0],
                 torch.max(y.pos, dim=0, keepdim=True)[0]]
        c     = [0.5 * (min[i] + max[i]) for i in [0, 1]]
        d     = [       max[i] - min[i]  for i in [0, 1]]
        s     = 1 + d[1] - d[0]
        y.pos = (y.pos - c[1]) * s + c[0]
        return y



