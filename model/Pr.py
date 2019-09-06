
single_view = torch.nn.Sequential(
                    VGGPerceptron((image_size,)*2,in_channels=5),
                    TensorDecisionLayer(512*(image_size//32)**2,size(P))
                )

model = torch.nn.Sequential(RenderLayer(renderer,camera),single_view,Batch_Sum())



# vx3xnxn
# vxnxn






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





class NormalMVSLoss(MVSLoss):
    def __init__(self,renderer,camera,name='NormalMVS',**kwargs):
        super(MVSLoss,self).__init__(renderer,camera,RenderLoss(),renderFcn=self.__renderFcn,name=name,**kwargs)

    def __renderFcn(self,input):
        out             = ?
        out[:,0:3,:,:]  = color2normal(out)
        camera          = repmat(torch.reshape(self.camera,(1,3,1,1)),(row(self.camera),1,out.shape[2],out.shape[3]))
        c               = torch.sum( out[:,0:3,:,:] * camera, 1, keepdim=True)
        c               = torch.where(c<=0,torch.ones_like(c),10*torch.ones_like(c))
        out[:,0:3,:,:] *= c
        return out




