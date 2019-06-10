from ACME.utility.nop           import *
from ACME.encoding.color2normal import *
from ACME.render.mesh2img       import *
from .Loss                      import *

class RenderLoss(Loss):
    def __init__(self,name='Render',renderFcn=None,**kwargs):
        super().__init__(name=name,**kwargs)
        self.renderFcn = renderFcn if renderFcn is not None else nop

    def __eval__(self,input,output):
        in_render  = self.renderFcn(input)
        out_render = self.renderFcn(output)
        return rgb_loss(in_render,out_render)+depth_loss(in_render,out_render)+alpha_loss(in_render,out_render)

    def rgb_loss(self,input,output):
        return torch.mean(torch.pow(rgb_channel(input)-rgb_channel(output),2)

    def depth_loss(self,input,output):
        return torch.mean(torch.pow(depth_channel(input)-depth_channel(output),2)

    def alpha_loss(self,input,output):
        return torch.sum(torch.pow(alpha_channel(input)-alpha_channel(output),2)




class NormalRenderLoss(RenderLoss):
    def __init__(self,name='NormalRender',**kwargs):
        super().__init__(name=name,**kwargs)

    def rgb_loss(self,input,output):
        N       = color2normal(rgb_channel(input))
        N_prime = color2normal(rgb_channel(output))
        c       = 0.5*(1-dot(N,N_prime,1)).squeeze()
        return torch.mean(torch.pow(c,2))




class MVSLoss(LossList):
    def __init__(self,renderer,camera,loss,renderFcn=None,name='MVS',**kwargs):
        super(MVSLoss,self).__init__(name=name,**kwargs)
        self.renderer  = renderer
        self.camera    = camera
        self.loss      = loss
        self.renderFcn = renderFcn if renderFcn is not None else nop

    def __eval__(self,input,output):
        in_mvs  = renderFcn(input)
        out_mvs = renderFcn(output)
        l = torch.zeros(1,dtype=torch.float,device=self.device)
        for i in row(self.camera):
            l += self.loss(in_mvs[i],out_mvs[i])
        return l
