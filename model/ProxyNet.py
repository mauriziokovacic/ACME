import torch

class PSignatureNet(torch.nn.Module):
    def __init__(self,views,image_size):
        super(SignatureNet,self).__init__()
        self.perceptive = torch.nn.Sequential(
                            VGGPerceptron((image_size,)*2,in_channels=5),
                            Flatten(),
                            DecisionLayer(512*(image_size//32)**2,[2048,1024]),
                            Batch_Flatten(),
                            DecisionLayer(views*1024,[2048,1024]),
                           )

    def forward(self,input):
        return torch.cat((self.perceptive(perceptive).flatten(),self.geometric(geometric).flatten()))

class PGSignatureNet(torch.nn.Module):
    def __init__(self,views,image_size):
        super(SignatureNet,self).__init__()
        self.perceptive = torch.nn.Sequential(
                            VGGPerceptron((image_size,)*2,in_channels=5),
                            Flatten(),
                            DecisionLayer(512*(image_size//32)**2,[2048,1024]),
                            Batch_Flatten(),
                            DecisionLayer(views*1024,[2048,1024]),
                           )
        dgcnn      = torch.load('DGCNN_weights/DGCNN_weights249.pkl').train(False)
        dgcnn.mlp2 = Identity()
        self.geometric  = DGCNNLayer(dgcnn)

    def forward(self,perceptive,geometric):
        return torch.cat((self.perceptive(perceptive).flatten(),self.geometric(geometric).flatten()))



class UVNet(torch.nn.Module):
    def __init__(self,renderer,camera,keep_output=False,attr='uv'):
        super(MyModel,self).__init__()
        self.keep     = keep_output
        self.attr     = attr
        self.render   = MVSRenderLayer(PositionRenderLayer(renderer),camera,keep_output=True,attr='mv')
        self.feature  = SignatureNet(row(camera),renderer.image_size)
        self.decision = torch.nn.Sequential(DecisionLayer(2048,[2048,1024,162*3]),Reshape((162,3)))
        self.sampler  = Sampler3D()


    def forward(self,input):
        mvs = self.render(input)
        uv  = self.feature(mvs,input)
        uv  = self.decision(uv)
        if self.keep:
            setattr(input,attr,uv)
        return self.sampler(mvs.permute(1,0,2,3),uv)




def narrow_loss(data):
    return torch.mean(data)+torch.var(data)



class PickLoss(Loss):
    def __init__(self,name='Pick',uvFcn=(lambda x:x.uv),**kwargs):
        super(PickLoss,self).__init__(name=name,**kwargs)
        self.uvFcn = uvFcn

    def __eval__(self,input,output):
        texture = input.mv.permute(1,0,2,3)
        uv      = self.uvFcn(output)
        return torch.sum(1-fetch_texture3D(texture,uv)[:,-1])



class PositionLoss(Loss):
    def __init__(self,name='Position',**kwargs):
        super(PositionLoss,self).__init__(**kwargs)

    def __eval__(self,input,output):
        texture = input.mv.permute(1,0,2,3)
        uv      = barycenter(output.uv, T=output.face)
        t       = barycenter(output.pos,T=output.face)
        return narrow_loss(sqnorm(t-color2position(fetch_texture3D(texture,uv)[:,0:3]))



class FaceLoss(LossList):
    def __init__(self,name='Face',**kwargs):
        super(FaceLoss,self).__init__(
            PickLoss(uvFcn=(lambda x:barycenter(x.uv,T=x.face))),
            PositionLoss(),
            name=name,**kwargs)





# Input is mesh
# Render it as positions
# Create uv
# Pick position from uv

# Position must be taken from within mask (mvs[:,-1,:,:]==1)

# Compute triangle uv
# Pick position from uv
# True triangle barycenter should be as close as possible to the picked
# Position must be taken from within mask


