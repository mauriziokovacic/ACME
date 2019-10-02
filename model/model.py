import warnings
import torch
from ..fileio.fileparts import *


class Model(torch.nn.Module):
    """
    A class representing a generic model architecture

    Attributes
    ----------
    name : str
        the name of the model
    device : str or torch.device
        the device to store the tensors to

    Methods
    -------
    freeze()
        freezes all the parameters in the model
    unfreeze()
        unfreezes all the parameters in the model
    save_model(path)
        stores the model state in the given path
    load_model(path)
        loads a model from the given path
    save_checkpoint(path)
        stores the model checkpoint in the given path
    load_checkpoint(path)
        loads the model checkpoint from the given path
    """

    def __init__(self, name='Model', device='cuda:0'):
        """
        Parameters
        ----------
        name : str (optional)
            the name of the model (default is 'Model')
        device : str or torch.device (optional)
            the device to store the tensors to (default is 'cuda:0')
        """

        super(Model, self).__init__()
        self.name   = name
        self.device = device
        
    def freeze(self):
        """
        Freezes all the parameters in the model
        
        Returns
        -------
        Model
            the model itself
        """
        
        for param in self.parameters():
            param.requires_grad = False
        return self
            
    def unfreeze(self):
        """
        Unfreezes all the parameters in the model
        
        Returns
        -------
        Model
            the model itself
        """
        
        for param in self.parameters():
            param.requires_grad = True
        return self

    def save_model(self, filename):
        """
        Stores the model state in the given filename.

        The file extension will be forced to '.pth'

        Parameters
        ----------
        filename : str
            the filename to store the model to

        Returns
        -------
        Model
            the model itself
        """

        path, file = fileparts(filename)[:2]
        torch.save(self, path + file + '.pth')
        return self

    def load_model(self, filename, strict=True):
        """
        Loads a model from the given filename

        The file extension will be forced to '.pth'

        Parameters
        ----------
        filename : str
            the filename to load the model from
        strict : bool (optional)
            if True loaded model must completely match with this. If False only compatible parameters will be loaded

        Returns
        -------
        Model
            the model itself
        """

        path, file = fileparts(filename)[:2]
        other = torch.load(path + file + '.pth', map_location=self.device)
        if strict and (not isinstance(other, self.__class__)):
            warnings.warn('Class type mismatch. Loading model failed', category=RuntimeWarning)
            return self
        self.load_state_dict(other.state_dict(), map_location=self.device, strict=strict)
        self.eval()
        return self

    def save_checkpoint(self, filename):
        """
        Stores the model checkpoint in the given filename

        The file extension will be forced to '.tar'

        Parameters
        ----------
        filename : str
            the filename to store the model checkpoint to

        Returns
        -------
        Model
            the model itself
        """

        path, file = fileparts(filename)[:2]
        torch.save({'model_state_dict': self.state_dict()}, path + file + '.tar')
        return self

    def load_checkpoint(self, filename, strict=True):
        """
        Loads the model checkpoint from the given filename

        The file extension will be forced to '.tar'

        Parameters
        ----------
        filename : str
            the filename to load the model checkpoint from
        strict : bool (optional)
            if True loaded checkpoint must completely match with the model parameters.
            If False only compatible parameters will be loaded

        Returns
        -------
        Model
            the model itself
        """

        path, file = fileparts(filename)[:2]
        checkpoint = torch.load(path + file + '.tar')
        self.load_state_dict(checkpoint['model_state_dict'], map_location=self.device, strict=strict)
        self.to(self.device)
        self.train()
        return self

    def __setattr__(self, key, value):
        self.__dict__[key] = value
        if key == 'device':
            self.to(device=value)
