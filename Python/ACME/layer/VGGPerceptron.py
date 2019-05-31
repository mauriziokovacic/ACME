import torch
from .Batch_Flatten import *

class VGGPerceptron(torch.nn.Module):
    def __init__(self,
             cfg=[64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
             in_channels=3,
             image_size=128,
             batch_norm=False,
            ):
        super(VGGPerceptron,self).__init__()
        self.__create_model(cfg,in_channels,image_size,batch_norm)

    def __create_model(self,cfg,in_channels,image_size,batch_norm):
        layers,pool = __create_layers(cfg,in_channels=in_channels,batch_norm=batch_norm)
        self.model = torch.nn.Sequential(
            *layers,
            torch.nn.AdaptiveAvgPool2d((image_size//(2**pool),)*2),
        )

    def __create_layers(self,cfg, in_channels=3,batch_norm=False):
        layers = []
        pool   = 0
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                pool   += 1
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1)]
                if batch_norm:
                    layers += [nn.BatchNorm2d(v)]
                layers += [nn.ReLU(inplace=True)]
                in_channels = v
        return layers,pool


    def forward(self,input):
        return self.model(input)



class MVSPerceptron(VGGPerceptron):
    def __init__(self,**kwargs):
        super(MVSPerceptron,self).__init__(**kwargs)
        self.model = torch.nn.Sequential(
            self.model,
            Batch_Flatten(),
        )
