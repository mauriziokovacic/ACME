from ...layer.Bypass        import *
from ...layer.HookLayer     import *
from ...layer.MeshUnpooling import *
from ...layer.ShapeLayer    import *
from .deformer_block        import *


class Pixel2Mesh(torch.nn.Module):
    """
    A class representing a module

    Attributes
    ----------

    Methods
    -------
    forward(x)
        evaluates the module
    """

    def __init__(self, proxy, vgg, cam=Camera()):
        """
        Parameters
        ----------
        """

        super(Pixel2Mesh, self).__init__()
        self.proxy    = ShapeLayer(proxy)
        self.vgg      = vgg
        self.hook     = HookLayer(Bypass(), [vgg[8], vgg[12], vgg[16]])
        self.unpool   = torch.nn.ModuleList([MeshUnpooling(), MeshUnpooling()])
        self.deformer = torch.nn.ModuleList([
            DeformerBlock(3 + 0 + (256 + 512 + 512), cam=cam),  # Proxy coordinates + previous features + pooled features
            DeformerBlock(3 + 128 + (256 + 512 + 512), cam=cam),  # Proxy coordinates + previous features + pooled features
            DeformerBlock(3 + 128 + (256 + 512 + 512), cam=cam),  # Proxy coordinates + previous features + pooled features
        ])
        self.add_modules('proxy', self.proxy)
        self.add_modules('vgg', self.vgg)
        self.add_modules('unpool', self.unpool)
        self.add_modules('deformer', self.deformer)

    def forward(self, x):
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

        y = self.proxy()
        _ = self.vgg(x.img)
        textures = self.hook()
        y, f = self.deformer[0](y, textures)
        y, f = self.unpool[0](y, f)
        y, f = self.deformer[1](y, textures, f=f)
        y, f = self.unpool[1](y, f)
        y, f = self.deformer[2](y, textures, f=f)
        return y
