import torch

class FeatureLayer(torch.nn.Module):
    def __init__(self,*feature):
        super(FeatureLayer,self).__init__()
        self.feature = torch.nn.ModuleList(*feature)

    def forward(self,input):
        out = [f(input) for f in self.feature]
        return torch.cat(out)
