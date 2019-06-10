import neural_renderer as nr
from ACME.math.unitvec import *

def toggle_lighting(renderer,status):
    """
    Toggles the directional lighting of the input renderer on or off depending on status

    Parameters
    ----------
    renderer : Neural Renderer
        an instance of a Neural Renderer
    status : bool
        the lighting status

    Returns
    -------
    None
    """

    if status:
        renderer.light_intensity_directional = 0.5
    else:
        renderer.light_intensity_directional = 0
    renderer.light_intensity_ambient = 1-renderer.light_intensity_directional



def enable_lighting(renderer):
    """
    Enables the lighting of the input renderer

    Parameters
    ----------
    renderer : Neural Renderer
        an instance of a Neural Renderer
    """

    toggle_lighting(renderer,True)



def disable_lighting(renderer):
    """
    Disables the lighting of the input renderer

    Parameters
    ----------
    renderer : Neural Renderer
        an instance of a Neural Renderer
    """

    toggle_lighting(renderer,False)



def create_renderer(camera_mode='look_at',image_size=256,lighting=False,device='cuda:0'):
    """
    Creates an instance of a Neural Renderer

    Parameters
    ----------
    camera_mode : str (optional)
        the camera mode of the renderer (default is 'look_at')
    image_size : int (optional)
        the output image size resolution (default is 256)
    lighting : bool (optional)
        if True turns of the directional lighting of the renderer (default is False)
    device : str or torch.device (optional)
        the tensor the renderer will be stored to (default is 'cuda:0')
    """

    R                 = nr.Renderer(camera_mode='look_at',image_size=image_size, device=device)
    R.eye             = unitvec(3,0,device=device)
    R.light_direction = -R.eye
    toggle_lighting(R,lighting)
    return R
