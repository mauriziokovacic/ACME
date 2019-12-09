import torch
from ...color.fetch_texture import *


class TexturePooling(torch.nn.Module):
    """
    A class implementing the texture pooling

    Methods
    -------
    forward(uv, *textures)
        fetches the textures features
    """

    def __init__(self):
        super(TexturePooling, self).__init__()

    def forward(self, uv, *textures):
        """
        Fetches the textures features

        Parameters
        ----------
        uv : Tensor
            the (N,2,) texture coordinates tensor
        textures : list
            a list of (M,...,) texture tensors to fetch

        Returns
        -------
        Tensor
            the (N,...,) output tensor
        """

        return torch.cat([fetch_texture2D(t, uv) for t in textures], dim=-1)
