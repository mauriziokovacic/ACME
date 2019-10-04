import os
import time
import torch
import warnings
from ..utility.nop import *


class Trainer(object):
    """
    A class representing a trainer object for an input architecture.

    Attributes
    ----------
    model : torch.nn.Module
        an architecture to train
    optimizer : torch.optim
        an optimizer
    loss : ACME.loss.Loss
        a loss function
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
    stateFcn : list
        a list of functions called after a single training iteration, sending the current training state

    Methods
    -------
    is_ready()
        returns whether or not the fundamentals attributes are set
    register_training_observer(observer)
        registers an observer to this trainer
    unregister_training_observer(observer)
        unregisters an observer from this trainer
    train(dataset,epochs,checkpoint,path,verbose)
        trains the model using the given dataset for the specified number of epochs, storing checkpoints into the given path
    test(input)
        tests an input against the model
    save_checkpoint(path)
        stores a checkpoint to a given path
    load_checkpoint(path)
        loads a checkpoint from a given path
    to(device)
        moves the trainer to the given device
    """

    def __init__(self,
                 model=None,
                 optimizer=None,
                 scheduler=None,
                 loss=None,
                 inputFcn=None,
                 outputFcn=None,
                 device='cuda:0',
                 name='Model',
                 ):
        self.model     = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss      = loss
        self.inputFcn  = inputFcn
        self.outputFcn = outputFcn
        self.stateFcn  = []
        self.device    = device
        self.name      = name
        self.epoch     = 0
        self.to(self.device)

    def is_ready(self):
        """
        Returns whether or not the fundamentals attributes are set

        Returns
        -------
        bool
            the trainer state
        """

        return (self.model is not None) and \
               (self.optimizer is not None) and \
               (self.loss is not None)

    def register_training_observer(self, observer):
        """
        Registers an observer to this trainer

        Parameters
        ----------
        observer : Training_Observer
            a training observer

        Returns
        -------
        Trainer
            the trainer itself
        """

        self.stateFcn += [observer.stateFcn]
        return self

    def unregister_training_observer(self, observer):
        """
        Unregisters an observer from this trainer

        Parameters
        ----------
        observer : Training_Observer
            a training observer

        Returns
        -------
        Trainer
            the trainer itself
        """

        self.stateFcn.remove(observer.stateFcn)
        return self

    def train(self,
              dataset,
              epochs=None,
              checkpoint=True,
              path=None,
              verbose=False):
        """
        Trains the model with the input dataset for a given number of epochs

        Parameters
        ----------
        dataset : DataLoader
            a dataloader object containing the dataset
        epochs : int (optional)
            the number of epochs to be performed. If None it will be automatically set to len(dataset) (default is None)
        checkpoint : bool (optional)
            if True stores a checkpoint of the training at the end of every epoch (default is True)
        path : str (optional)
            the path where to store checkpoints and final model. If None it will be set to the current working directory (default is None)
        verbose : bool (optional)
            if True print debug messages to the console (default is False)
        """

        if not self.is_ready():
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
                print('Epoch: {}...'.format(self.epoch), end='')
            for i, input in enumerate(dataset):
                t    = time.time()
                x    = inputFcn(input)
                y    = outputFcn(self.model(x))
                loss = self.loss.eval(x, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(loss)
                    else:
                        self.scheduler.step()
                for fcn in self.stateFcn:
                    fcn(
                        input=x,
                        output=y,
                        loss=self.loss.to_dict(),
                        epoch=(e, self.epoch, epochs),
                        iteration=(i, n),
                        t=time.time()-t
                    )
            if verbose:
                print('DONE')
            if checkpoint:
                self.save_checkpoint(path)
        return

    def test(self, dataset, verbose=False):
        """
        Tests an input against the current model

        Parameters
        ----------
        dataset : DataLoader
            a dataloader object containing the dataset
        verbose : bool (optional)
            if True print debug messages to the console (default is False)

        Returns
        -------
        list
            a list containing a (input, output, loss) tuple for each dataset entry
        """

        if not self.is_ready():
            warnings.warn('Trainer is not ready. Set properly model, optimizer and loss.', RuntimeWarning)
            return

        n = len(dataset)
        inputFcn  = self.inputFcn
        if inputFcn is None:
            inputFcn = nop
        outputFcn = self.outputFcn
        if outputFcn is None:
            outputFcn = nop

        out = []
        self.model.eval()
        if verbose:
            print('Starting test...')
        for i, input in enumerate(dataset):
            if verbose:
                print('Object: {}...'.format(i), end='')
            t = time.time()
            x = inputFcn(input)
            y = outputFcn(self.model(x))
            l = self.loss.eval(x, y)
            out += [(x, y, l)]
            if verbose:
                print('DONE')
        if verbose:
            print('TEST DONE')
        return out

    def save_checkpoint(self, path=None):
        """
        Stores a checkpoint in the given path

        Parameters
        ----------
        path : str (optional)
            the path to store the data to. If None it will be set to the current working directory (default is None)
        """

        if not self.is_ready():
            warnings.warn('Trainer is not ready. Set properly model, optimizer and loss.', RuntimeWarning)
            return
        if path is None:
            path = os.getcwd()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
            'loss': self.loss.value.item(),
            'epoch': self.epoch,
        }, path + '/' + self.name + '.tar')

    def load_checkpoint(self, path=None):
        """
        Loads a checkpoint from the given path

        Parameters
        ----------
        path : str (optional)
            the path to load the data from. If None it will be set to the current working directory (default is None)
        """

        if not self.is_ready():
            warnings.warn('Trainer is not ready. Set properly model, optimizer and loss.', RuntimeWarning)
            return
        if path is None:
            path = os.getcwd() + '/' + self.name + '.tar'
        if not os.path.isfile(path):
            print('File ' + path + ' does not exists.')
            return
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'], map_location=self.device)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.loss.value = checkpoint['loss']
        self.epoch      = checkpoint['epoch']
        self.model.to(self.device)
        self.model.train()

    def to(self, device):
        """
        Moves the trainer to the given device

        Parameters
        ----------
        device : str or torch.device
            the device to store the tensors to

        Returns
        -------
        Trainer
            the trainer itself
        """

        self.device = device
        return self

    def __setattr__(self, key, value):
        self.__dict__[key] = value
        if key == 'device':
            if hasattr(self, 'model') :
                if self.model is not None:
                    self.model.to(self.device)
            if hasattr(self, 'loss'):
                if self.loss is not None:
                    self.loss.to(self.device)

    def __call__(self,
                 dataset,
                 epochs=None,
                 checkpoint=True,
                 path=None,
                 verbose=False):
        self.train(dataset, epochs=epochs, checkpoint=checkpoint, path=path, verbose=verbose)
