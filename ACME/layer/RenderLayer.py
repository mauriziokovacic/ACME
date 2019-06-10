import torch
from ACME.utility.nop     import *
from ACME.render.mesh2img import *

class RenderLayer(torch.nn.Module):
    """
    A class representing a rendering layer

    Attributes
    ----------
    renderer : Neural Renderer
        an instance of a Neural renderer
    postFcn : callable
        a post processing function applied to the output image
    keep : bool
        if True keeps the output in an attribute of the input data
    attr : str
        the name of the attribute to store the output to

    Methods
    -------
    __colorFcn(input)
        creates colors from the input data
    forward(input)
        returns the rendered input
    """

    def __init__(self,renderer,postFcn=nop,keep_output=False,attr='img'):
        """
        Parameters
        ----------
        renderer : Neural Renderer
            an instance of a Neural Renderer
        postFcn : callable (optional)
            a post processing function applied to the output image (default is nop)
        keep_output : bool (optional)
            if True keeps the output in an attribute of the input data (default is False)
        attr : str (optional)
            the name of the attribute to store the output to (default is 'img')
        """

        super(RenderLayer,self).__init__()
        self.renderer = renderer
        self.postFcn  = postFcn
        self.keep     = keep_output
        self.attr     = attr



    def __colorFcn(self,input):
        """
        Creates colors from the input data

        Parameters
        ----------
        input : Data object
            the input data

        Returns
        -------
        Tensor
            the color tensor
        """

        return None



    def forward(self,input):
        """
        Returns the rendered input data

        Parameters
        ----------
        input : Data object
            the input data

        Returns
        -------
        Tensor
            the rendered input
        """

        P   = input.pos
        T   = input.face
        out = mesh2img(self.renderer,T,P,self.__colorFcn(input),postFcn=self.postFcn)
        if self.keep:
            setattr(input,self.attr,out)
        return out



class NormalRenderLayer(RenderLayer):
    """
    A class representing a rendering layer, rendering normals

    Attributes
    ----------
    renderer : Neural Renderer
        an instance of a Neural renderer
    postFcn : callable
        a post processing function applied to the output image
    keep : bool
        if True keeps the output in an attribute of the input data
    attr : str
        the name of the attribute to store the output to
    per_vertex : bool
        if True uses the per vertex normals, face normals otherwise

    Methods
    -------
    __colorFcn(input)
        creates colors from the input data
    forward(input)
        returns the rendered input
    """

    def __init__(self,renderer,per_vertex=True,**kwargs):
        """
        Parameters
        ----------
        renderer : Neural Renderer
            an instance of a Neural Renderer
        per_vertex : bool (optional)
            if True uses the per vertex normals, face normals otherwise
        **kwargs
            the keyword arguments of RenderLayer
        """

        super(NormalRenderLayer,self).__init__(renderer,**kwargs)
        self.per_vertex = per_vertex



    def __colorFcn(self,input):
        """
        Creates colors from the normals of the input data

        If the input already has an attribute called 'normals', that
        data will be used, otherwise it computes the normals

        Parameters
        ----------
        input : Data object
            the input data

        Returns
        -------
        Tensor
            the color tensor
        """

        if hasattr(input,'normals'):
            return normal2color(input.normals)
        if per_vertex:
            return normal2color(vertex_normal(input.pos,input.face))
        return normal2color(triangle_normal(input.pos,input.face))



class PositionRenderLayer(RenderLayer):
    """
    A class representing a rendering layer, rendering positions

    Attributes
    ----------
    renderer : Neural Renderer
        an instance of a Neural renderer
    postFcn : callable
        a post processing function applied to the output image
    keep : bool
        if True keeps the output in an attribute of the input data
    attr : str
        the name of the attribute to store the output to
    per_vertex : bool
        if True uses the per vertex normals, face normals otherwise

    Methods
    -------
    __colorFcn(input)
        creates colors from the input data
    forward(input)
        returns the rendered input
    """


    def __init__(self,renderer,**kwargs):
        """
        Parameters
        ----------
        renderer : Neural Renderer
            an instance of a Neural Renderer
        **kwargs
            the keyword arguments of RenderLayer
        """

        super(PositionRenderLayer,self).__init__(renderer,**kwargs)



    def __colorFcn(self,input):
        """
        Creates colors from the positions of the input data

        The posistions are intended to be in range [-1,1]

        Parameters
        ----------
        input : Data object
            the input data

        Returns
        -------
        Tensor
            the color tensor
        """

        return position2color(input.pos,min=-1,max=1)



class MVSRenderLayer(torch.nn.Module):
    """
    A class representing a multi view stack rendering layer

    Attributes
    ----------
    layer : RenderLayer
        a rendering layer
    camera : Tensor
        the positions of the camera
    keep : bool
        if True keeps the output in an attribute of the input data
    attr : str
        the name of the attribute to store the output to

    Methods
    -------
    __render(input,camera)
        renders the input from a specific camera position
    forward(input)
        returns the rendered multi view stack of the input data
    """


    def __init__(self,render_layer,camera,keep_output=False,attr='mvs'):
        """
        Parameters
        ----------
        render_layer : RenderLayer
            a rendering layer
        camera : Tensor
            the positions of the camera
        keep_output : bool (optional)
            if True keeps the output in an attribute of the input data (default is False)
        attr : str (optional)
            the name of the attribute to store the output to (default is 'mvs')
        """

        super(MVSRenderLayer,self).__init__()
        self.layer  = render_layer
        self.camera = camera
        self.keep   = keep_output
        self.attr   = attr



    def __render(self,input,camera):
        """
        Renders the input from a specific camera position

        Parameters
        ----------
        input : Data object
            the input data
        camera : Tensor
            the position of the camera

        Returns
        -------
        Tensor
            an image tensor
        """

        self.layer.renderer.eye             =  camera
        self.layer.renderer.light_direction = -camera
        return self.layer(input)



    def forward(self,input):
        """
        Returns the multi view stack of the input data

        Parameters
        ----------
        input : Data object
            the input data

        Returns
        -------
        Tensor
            the multi view stack image tensor
        """

        out = torch.cat(tuple(self.__render(input,c).unsqueeze(0) for c in self.camera),dim=0)
        if self.keep:
            setattr(input,self.attr,out)
        return out
