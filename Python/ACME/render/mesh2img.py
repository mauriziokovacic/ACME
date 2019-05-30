import torch
from utility.col        import *
from utility.IntTensor  import *
from utility.repmat     import *
from utility.nop        import *
from topology.ispoly    import *
from topology.poly2poly import *
from.color2nr           import *



def mesh2img(renderer,T,P,C=None,postFcn=nop):
    """
    Renders an input mesh with the given renderer

    Parameters
    ----------
    renderer : Neural Renderer
        an instance of the Neural Renderer
    T : LongTensor
        the topology tensor
    P : Tensor
        the points set tensor
    C : Tensor or Uint8Tensor (optional)
        the RGB color tensor. If None the mesh will be colored white (default is None)
    postFcn : callable (optional)
        a function to be applied to the Neural Renderer output (defalut is nop)

    Returns
    -------
    Tensor
        the Neural Renderer image in RGBDA format if postFcn is nop
    """

    t = T.clone()
    if C is None:
        c = torch.ones(col(T),3,dtype=torch.float,device=renderer.device)
    else:
        c = C
    if isquad(t):
        t = quad2tri(t)
        c = repmat(c.clone(),(2,1))
    c = color2nr(t,c,texture_size=2,dtype=torch.float32,device=renderer.device)
    t = IntTensor(torch.t(t),device=renderer.device)
    p = P[None,:,:]
    t = t[None,:,:]
    I = renderer(p,t,c)
    I = postFcn(torch.cat((I[0],
                           1-normalize(I[1].unsqueeze(1),min=renderer.near,max=renderer.far),
                           I[2].unsqueeze(1)), dim=1).to(device=renderer.device))
    return I



def image_channel(I,channel):
    """
    Extracts the specified channel(s) from the input Neural Renderer image

    Parameters
    ----------
    I : Tensor
        the Neural Renderer image
    channel : int or list of ints
        the channel(s) to extract

    Returns
    -------
    Tensor
        the selected channel(s)
    """

    return I[:,channel,:,:]



def red_channel(I):
    """
    Extracts the red channel from the input Neural Renderer image

    Parameters
    ----------
    I : Tensor
        the Neural Renderer image

    Returns
    -------
    Tensor
        the red channel
    """

    return image_channel(I,0)



def green_channel(I):
    """
    Extracts the green channel from the input Neural Renderer image

    Parameters
    ----------
    I : Tensor
        the Neural Renderer image

    Returns
    -------
    Tensor
        the green channel
    """

    return image_channel(I,1)



def blue_channel(I):
    """
    Extracts the blue channel from the input Neural Renderer image

    Parameters
    ----------
    I : Tensor
        the Neural Renderer image

    Returns
    -------
    Tensor
        the blue channel
    """

    return image_channel(I,2)



def rgb_channel(I):
    """
    Extracts the RGB channels from the input Neural Renderer image

    Parameters
    ----------
    I : Tensor
        the Neural Renderer image

    Returns
    -------
    Tensor
        the RGB channels
    """

    return image_channel(I,(0,1,2))



def depth_channel(I):
    """
    Extracts the depth channel from the input Neural Renderer image

    Parameters
    ----------
    I : Tensor
        the Neural Renderer image

    Returns
    -------
    Tensor
        the depth channel
    """

    return image_channel(I,3)



def rgbd_channel(I):
    """
    Extracts the RGBD channels from the input Neural Renderer image

    Parameters
    ----------
    I : Tensor
        the Neural Renderer image

    Returns
    -------
    Tensor
        the RGBD channels
    """

    return image_channel(I,(0,1,2,3,4))



def alpha_channel(I):
    """
    Extracts the alpha channel from the input Neural Renderer image

    Parameters
    ----------
    I : Tensor
        the Neural Renderer image

    Returns
    -------
    Tensor
        the alpha channel
    """

    return image_channel(I,4)




def mesh2rgb(renderer,T,P,C=None):
    """
    Renders the RGB channels of an input mesh with the given renderer

    Parameters
    ----------
    renderer : Neural Renderer
        an instance of the Neural Renderer
    T : LongTensor
        the topology tensor
    P : Tensor
        the points set tensor
    C : Tensor or Uint8Tensor (optional)
        the RGB color tensor. If None the mesh will be colored white (default is None)

    Returns
    -------
    Tensor
        the Neural Renderer image in RGB format
    """
    return mesh2img(renderer,T,P,C=C,postFcn=rgb_channel)



def mesh2depth(renderer,T,P,C=None):
    """
    Renders the depth channel of an input mesh with the given renderer

    Parameters
    ----------
    renderer : Neural Renderer
        an instance of the Neural Renderer
    T : LongTensor
        the topology tensor
    P : Tensor
        the points set tensor
    C : Tensor or Uint8Tensor (optional)
        the RGB color tensor. If None the mesh will be colored white (default is None)

    Returns
    -------
    Tensor
        the Neural Renderer image in D format
    """
    return mesh2img(renderer,T,P,C=C,postFcn=depth_channel)



def mesh2rgbd(renderer,T,P,C=None):
    """
    Renders the RGBD channels of an input mesh with the given renderer

    Parameters
    ----------
    renderer : Neural Renderer
        an instance of the Neural Renderer
    T : LongTensor
        the topology tensor
    P : Tensor
        the points set tensor
    C : Tensor or Uint8Tensor (optional)
        the RGB color tensor. If None the mesh will be colored white (default is None)

    Returns
    -------
    Tensor
        the Neural Renderer image in RGBD format
    """
    return mesh2img(renderer,T,P,C=C,postFcn=rgbd_channel)