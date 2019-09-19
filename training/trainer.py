import os
import time
import torch
import warnings
from ..utility.istuple import *
from ..utility.nop     import *


class Trainer(object):
    """
    A class representing a trainer object for an input architecture.

    Attributes
    ----------
    model : torch.nn.Module
        an architecture to train
    optimizer : torch.optim
        an optimizer
    loss : ACME.loss.GlobalLoss
        a global loss function
    device : str or torch.device
        the device the tensors will be stored to
    epoch : int
        the current training epoch
    name : str
        the model name
    inputFcn : callable
        a preprocessing function over the architecture input
    outputFcn : callable
        a postprocessing function over the architecture output
    stateFcn : callable
        a function called after a single training iteration, sending the current training state

    Methods
    -------
    isready()
        returns whether or not the fundamentals attributes are set
    train(dataset,epochs,checkpoint,finalNetwork,path,verbose)
        trains the model using the given dataset for the specified number of epochs, storing checkpoints and final model into a given path
    test(input)
        tests an input against the model
    save_checkpoint(path)
        stores a checkpoint to a given path
    load_checkpoint(path)
        loads a checkpoint from a given path
    save_model(path)
        stores the trained model to the given path
    load_model(path)
        loads the trained model from the given path
    """

    def __init__(self,
                 model=None,
                 optimizer=None,
                 scheduler=None,
                 loss=None,
                 name='Model',
                 device='cuda:0',
                 inputFcn=None,
                 outputFcn=None,
                 stateFcn=None):
        self.model     = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss      = loss
        self.device    = device
        self.epoch     = 0
        self.name      = name
        self.inputFcn  = inputFcn
        self.outputFcn = outputFcn
        self.stateFcn  = stateFcn

    def isready(self):
        """
        Returns whether or not the fundamentals attributes are set

        Returns
        -------
        bool
            the trainer state
        """

        return (self.model is not None) and\
               (self.optimizer is not None) and\
               (self.loss is not None)
               #(self.scheduler is not None) and\

    def train(self,
              dataset,
              epochs=None,
              checkpoint=True,
              finalNetwork=True,
              path=None,
              verbose=False):
        """
        Trains the model with the input dataset for a given number of epochs

        Parameters
        ----------
        dataset :
            a dataloader object containing the dataset
        epochs : int (optional)
            the number of epochs to be performed. If None it will be automatically set to len(dataset) (default is None)
        checkpoint : bool (optional)
            if True stores a checkpoint of the training at the end of every epoch (default is True)
        finalNetwork : bool (optional)
            if True stores the final trained model (default is True)
        path : str (optional)
            the path where to store checkpoints and final model. If None it will be set to the current working directory (default is None)
        verbose : bool (optional)
            if True print debug messages to the console (default is False)
        """

        if not self.isready():
            warnings.warn('Trainer is not ready. Set properly model, optimizer and loss.', RuntimeWarning)
            return
        n = len(dataset)
        if path is None:
            path = os.getcwd()
        if epochs is None:
            epochs = n
        e = self.epoch

        inputFcn  = self.inputFcn
        if inputFcn is None:
            inputFcn = nop
        outputFcn = self.outputFcn
        if outputFcn is None:
            outputFcn = nop
        self.model.train()
        self.model.zero_grad()
        for self.epoch in range(e, epochs):
            if verbose:
                print('Epoch:' + str(self.epoch) + '...', end='')
            i = 0
            for input in dataset:
                t    = time.time()
                x    = inputFcn(input)
                y    = outputFcn(self.model(x))
                loss = self.loss.eval(x, *y if istuple(y) else y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(loss)
                    else:
                        self.scheduler.step()
                if self.stateFcn is not None:
                    self.stateFcn(x,
                                  y,
                                  self.loss.to_dict(),
                                  epoch=(e, self.epoch, epochs),
                                  iteration=(i, n),
                                  t=time.time()-t)
                i += 1
            if verbose:
                print('DONE')
            if checkpoint:
                self.save_checkpoint(path)
        if finalNetwork:
            self.save_model(path)

    def test(self, input):
        """
        Tests an input against the current model

        Parameters
        ----------
        input : object
            an input object

        Returns
        -------
        object
            the model output
        """

        if not self.isready():
            warnings.warn('Trainer is not ready. Set properly model, optimizer and loss.', RuntimeWarning)
            return

        inputFcn  = self.inputFcn
        if inputFcn is None:
            inputFcn = nop
        outputFcn = self.outputFcn
        if outputFcn is None:
            outputFcn = nop

        self.model.eval()
        output = outputFcn(self.model(inputFcn(input)))
        return output

    def save_checkpoint(self, path=None):
        """
        Stores a checkpoint in the given path

        Parameters
        ----------
        path : str (optional)
            the path to store the data to. If None it will be set to the current working directory (default is None)
        """

        if not self.isready():
            warnings.warn('Trainer is not ready. Set properly model, optimizer and loss.', RuntimeWarning)
            return
        if path is None:
            path = os.getcwd()
        path = path + '/' + self.name + '.tar'
        torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else '',
                    'loss': self.loss.value.item(),
                    'epoch': self.epoch,
                    }, path)

    def load_checkpoint(self, path=None):
        """
        Loads a checkpoint from the given path

        Parameters
        ----------
        path : str (optional)
            the path to load the data from. If None it will be set to the current working directory (default is None)
        """

        if not self.isready():
            warnings.warn('Trainer is not ready. Set properly model, optimizer and loss.', RuntimeWarning)
            return
        if path is None:
            path = os.getcwd()
            path = path + '/' + self.name + '.tar'
        if not os.path.isfile(path):
            print('File ' + path + ' does not exists.')
            return
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])#,map_location=self.device)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.loss.value = checkpoint['loss']
        self.epoch      = checkpoint['epoch']
        self.model.train()

    def save_model(self, path=None):
        """
        Stores the final model state in the given path

        Parameters
        ----------
        path : str (optional)
            the path to store the model to. If None it will be set to the current working directory (default is None)
        """

        if not self.isready():
            warnings.warn('Trainer is not ready. Set properly model, optimizer and loss.', RuntimeWarning)
            return
        if path is None:
            path = os.getcwd()
            path = path + '/' + self.name + '.pth'
        torch.save(self.model, path)

    def load_model(self, path=None):
        """
        Loads a final model from the given path

        Parameters
        ----------
        path : str (optional)
            the path to load the model from. If None it will be set to the current working directory (default is None)
        """

        if not self.isready():
            warnings.warn('Trainer is not ready. Set properly model, optimizer and loss.', RuntimeWarning)
            return
        if path is None:
            path = os.getcwd()
            path = path + '/' + self.name + '.pth'
        if not os.path.isfile(path):
            print('File ' + path + ' does not exists.')
            return
        self.model.load_state_dict(torch.load(path,map_location=self.device))
        self.model.to(device=self.device)
        self.model.eval()

    def __call__(self,
                 dataset,
                 epochs=None,
                 checkpoint=True,
                 finalNetwork=True,
                 path=None):
        self.train(dataset, epochs=epochs, checkpoint=checkpoint, finalNetwork=finalNetwork, path=path)
