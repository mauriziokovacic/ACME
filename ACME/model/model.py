import os
import warnings
import torch



class Model(torch.nn.Module):
    """
    A class representing a generic model architecture

    Attributes
    ----------
    name : str
        the name of the model

    Methods
    -------
    save_model(path)
        stores the model state in the given path
    load_model(path)
        loads a model form the given path
    """

    def __init__(self,name='Model'):
        """
        Parameters
        ----------
        name : str (optional)
            the name of the model (default is 'Model')
        """

        super(Model,self).__init__()
        self.name  = name



    def save_model(self,path=None):
        """
        Stores the model state in the given path

        Parameters
        ----------
        path : str (optional)
            the path to store the model to. If None it will be set to the current working directory (default is None)
        """

        if path is None:
            path = os.getcwd()
            path = path + '/' + self.name + '.pth'
        torch.save(self, path)
        return self



    def load_model(self,path=None):
        """
        Loads a model from the given path

        Parameters
        ----------
        path : str (optional)
            the path to load the model from. If None it will be set to the current working directory (default is None)
        """

        if path is None:
            path = os.getcwd()
            path = path + '/' + self.name + '.pth'
        if not os.path.isfile(path):
            warnings.warn('File ' + path + ' does not exists.')
            return
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        return self
