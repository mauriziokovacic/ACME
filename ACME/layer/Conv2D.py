import torch

class Conv2D(torch.nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=3, padding=1):
        super(Conv2D,self).__init__()
        self.model = torch.nn.Sequential(
                        torch.nn.Conv2d(out_channels,out_channels,kernel_size=(kernel_size,1),padding=padding),
                        torch.nn.Conv2d( in_channels,out_channels,kernel_size=(1,kernel_size),padding=padding),
                     )

    def forward(self,input):
        return self.model(input)
