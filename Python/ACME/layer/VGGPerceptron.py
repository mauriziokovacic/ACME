import torch
from .Batch_Flatten import *

class VGGPerceptron(torch.nn.Module):
    def __init__(self,
             *data_size,
             cfg=[64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
             in_channels=3,
             batch_norm=False,
            ):
        super(VGGPerceptron,self).__init__()
        self.__create_model(cfg,in_channels,*data_size,batch_norm)

    def __create_model(self,cfg,in_channels,data_size,batch_norm):
        layers,pool = self.__create_layers(cfg,in_channels=in_channels,batch_norm=batch_norm,dim=len(data_size))
        self.model = torch.nn.Sequential(
            *layers,
            eval('torch.nn.AdaptiveAvgPool'+str(len(data_size))+'d('+(str(tuple((eval(str(x)+'//'+str(2**pool))) for x in data_size)))+')')
        )

    def __create_layers(self,cfg, in_channels=3,batch_norm=False,dim=2):
        layers = []
        pool   = 0
        for v in cfg:
            if v == 'M':
                layers += [eval('torch.nn.MaxPool'+str(dim)+'d(kernel_size=2, stride=2)')]
                pool   += 1
            else:
                layers += [eval('torch.nn.Conv'+str(dim)+'d(in_channels, v, kernel_size=3, padding=1)')]
                if batch_norm:
                    layers += [eval('torch.nn.BatchNorm'+str(dim)+'d(v)')]
                layers += [torch.nn.ReLU(inplace=True)]
                in_channels = v
        return layers,pool


    def forward(self,input):
        return self.model(input)




class MVSPerceptron(VGGPerceptron):
    def __init__(self,*args,**kwargs):
        super(MVSPerceptron,self).__init__(**kwargs)
        self.model = torch.nn.Sequential(
            self.model,
            Batch_Flatten(),
        )
