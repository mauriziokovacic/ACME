from ..utility.ACMEClass import *
from ..utility.islist    import *


class TrainerObserver(ACMEClass):
    """
    A class representing an object to observe the state of a trainer while training.

    Methods
    -------
    bind(trainer)
        binds the Training Observer to a given trainer
    unbind(trainer)
        unbinds the Training Observer to a given trainer
    stateFcn(input,output,loss,epoch,iteration,t)
        Receives the trainer state and executes the routine functions
    iterationFcn(input,output,loss,epoch,iteration,t)
        Receives the trainer state at the end of each iteration
    epochFcn(input,output,loss,epoch,iteration,t)
        Receives the trainer state at the end of each epoch
    endFcn(input,output,loss,epoch,iteration,t)
        Receives the trainer state at the end of the training
    """

    def __init__(self, trainers=None):
        """
        Parameters
        ----------
        trainers : Trainer object or list (optional)
            If not None, binds the Training State Manager to the given trainer(s) (default is None)
        """

        super(TrainerObserver, self).__init__()
        if trainers is not None:
            self.bind(trainers)

    def bind(self, trainers):
        """
        Binds the Training Observer to a given trainer

        Parameters
        ----------
        trainers : Trainer or list
            a trainer object or list of trainer objects

        Returns
        -------
        TrainingObserver
            the training observer itself
        """

        if islist(trainers):
            for t in trainers:
                t.register_training_observer(self)
        else:
            trainers.register_training_observer(self)
        return self

    def unbind(self, trainers):
        """
        Unbinds the Training Observer to a given trainer

        Parameters
        ----------
        trainers : Trainer or list
            a trainer object or list of trainer objects

        Returns
        -------
        TrainingObserver
            the training observer itself
        """

        if islist(trainers):
            for t in trainers:
                t.unregister_training_observer(self)
        else:
            trainers.unregister_training_observer(self)
        return self

    def stateFcn(self, name, model, input, output, loss, epoch, iteration, t):
        """
        Receives the trainer state and executes the routine functions

        Parameters
        ----------
        name : str
            the trainer name
        model : torch.nn.Module
            the training model
        input : object
            the architecture input
        output : object
            the architecture output
        loss : dict
            a dictionary containing the losses names and values
        epoch : tuple
            a tuple containing the current epoch and the max epoch
        iteration : tuple
            a tuple containing the current iteration and the max iteration within an epoch
        t : float
            the execution time of last iteration

        Returns
        -------
        None
        """

        e = epoch
        i = iteration
        g = (e[0] * i[1] + i[0], e[1] * i[1])
        self.iterationFcn(name, model, input, output, loss, epoch, iteration, t)
        if ((g[0] + 1) % i[1]) == 0:
            self.epochFcn(name, model, input, output, loss, epoch, iteration, t)
        if g[0] == (g[1] - 1):
            self.endFcn(name, model, input, output, loss, epoch, iteration, t)
        return

    def iterationFcn(self, name, model, input, output, loss, epoch, iteration, t):
        """
        Receives the trainer state at the end of each iteration

        Parameters
        ----------
        name : str
            the trainer name
        model : torch.nn.Module
            the training model
        input : object
            the architecture input
        output : object
            the architecture output
        loss : dict
            a dictionary containing the losses names and values
        epoch : tuple
            a tuple containing the current epoch and the max epoch
        iteration : tuple
            a tuple containing the current iteration and the max iteration within an epoch
        t : float
            the execution time of last iteration

        Returns
        -------
        None
        """

        return

    def epochFcn(self, name, model, input, output, loss, epoch, iteration, t):
        """
        Receives the trainer state at end of each epoch

        Parameters
        ----------
        name : str
            the trainer name
        model : torch.nn.Module
            the training model
        input : object
            the architecture input
        output : object
            the architecture output
        loss : dict
            a dictionary containing the losses names and values
        epoch : tuple
            a tuple containing the current epoch and the max epoch
        iteration : tuple
            a tuple containing the current iteration and the max iteration within an epoch
        t : float
            the execution time of last iteration

        Returns
        -------
        None
        """

        return

    def endFcn(self, name, model, input, output, loss, epoch, iteration, t):
        """
        Receives the trainer state at end of the training

        Parameters
        ----------
        name : str
            the trainer name
        model : torch.nn.Module
            the training model
        input : object
            the architecture input
        output : object
            the architecture output
        loss : dict
            a dictionary containing the losses names and values
        epoch : tuple
            a tuple containing the current epoch and the max epoch
        iteration : tuple
            a tuple containing the current iteration and the max iteration within an epoch
        t : float
            the execution time of last iteration

        Returns
        -------
        None
        """

        return
