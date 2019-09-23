import neural_renderer as nr
from ..math.unitvec import *


class Renderer(nr.Renderer):
    """
    A class extending the Neural Renderer

    Attributes
    ----------
    device : str or torch.device (optional)
        the tensor the renderer will be stored to (default is 'cuda:0')
    culling : str (optional)
        the current active face culling (default is None)

    Methods
    -------
    toggle_lighting(status)
        toggles the renderer directional lighting on or off depending on status
    enable_lighting()
        enables the renderer lighting
    disable_lighting()
        disables the renderer lighting
    disable_culling()
        disables face culling
    enable_back_culling()
        enables back face culling
    enable_front_culling()
        enables front face culling
    """

    def __init__(self, device='cuda:0', culling=None, lighting=False, **kwargs):
        """
        Parameters
        ----------
        device : str or torch.device (optional)
            the device the tensors will be stored to (default is 'cuda:0')
        culling : str (optional)
            the current active face culling, either 'front' or 'back'.
            If None no culling is performed (default is None)
        lighting : bool (optional)
            if True activates the lighting, False otherwise
        **kwargs : ...
            the Neural Renderer keyworded arguments
        """

        super(Renderer, self).__init__(camera_mode='look_at', **kwargs)
        self.eye             = unitvec(3, 0, device=device)
        self.light_direction = -self.eye
        self.device          = device
        self.culling         = culling
        self.toggle_lighting(lighting)

    def toggle_lighting(self, status):
        """
        Toggles the renderer directional lighting on or off depending on status

        Parameters
        ----------
        status : bool
            the lighting status
        """

        if status:
            self.light_intensity_directional = 0.5
        else:
            self.light_intensity_directional = 0
        self.light_intensity_ambient = 1-self.light_intensity_directional

    def enable_lighting(self):
        """Enables the renderer lighting"""

        self.toggle_lighting(True)

    def disable_lighting(self):
        """Disables the renderer lighting"""

        self.toggle_lighting(False)

    def disable_culling(self):
        """Disables face culling"""

        self.culling = None

    def enable_back_culling(self):
        """
        Enables back face culling
        """

        self.culling = 'back'

    def enable_front_culling(self):
        """
        Enables front face culling
        """

        self.culling = 'front'
