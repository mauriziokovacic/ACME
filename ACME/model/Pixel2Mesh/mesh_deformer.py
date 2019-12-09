from torch_geometric   import GCNConv
from ...layer.G_ResNet import *


class MeshDeformer(torch.nn.Module):
    def __init__(self, in_channels):
        super(MeshDeformer, self).__init__()
        self.add_module('resnet', G_ResNet(GCNConv,
                                           in_channels,
                                           [128, ]*14,
                                           adjacency={2: [0, 1],
                                                      5: [3, 4],
                                                      8: [6, 7],
                                                      11: [9, 10]}))
        self.add_module('coord', GCNConv(128, 3))

    def forward(self, x, edge_index, *args, **kwargs):
        f = self.resnet(x, edge_index, **kwargs)
        c = self.coord(f, edge_index, *args, **kwargs)
        return f, c


