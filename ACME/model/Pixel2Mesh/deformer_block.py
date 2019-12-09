from .mesh_deformer import *
from .projector     import *
from .texture_pool  import *


class DeformerBlock(torch.nn.Module):
    """
    A class representing a module

    Attributes
    ----------

    Methods
    -------
    forward(x)
        evaluates the module
    """

    def __init__(self, in_channels, cam=Camera()):
        """
        Parameters
        ----------
        """

        super(DeformerBlock, self).__init__()
        self.proj = Projector(cam=cam)
        self.pool = TexturePooling()
        self.mlp  = MeshDeformer(in_channels)
        self.add_module('mlp', self.mlp)

    def forward(self, x, *textures, f=None):
        """
        Evaluates the module

        Parameters
        ----------
        x : Tensor
            the input tensor

        Returns
        -------
        Tensor
            the output tensor
        """

        uv = self.proj(x.pos)
        s  = self.pool(uv, *textures)
        if f is not None:
            s = torch.cat((s, f), dim=-1)
        x.pos, s = self.mlp(torch.cat((x.pos, s), dim=-1), x.edge_index)
        return x, s
