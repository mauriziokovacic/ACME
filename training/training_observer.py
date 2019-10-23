class Training_Observer(object):
    """
    A class representing an object to observe the state of a trainer while training.

    Attributes
    ----------
    epoch : int
        the last epoch processed
    iter : int
        the last iteration processed

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

    def __init__(self, trainer=None):
        """
        Parameters
        ----------
        trainer : Trainer object (optional)
            If not None, binds the Training State Manager to the given trainer (default is None)
        """

        self.epoch = None
        self.iter  = None
        if trainer is not None:
            self.bind(trainer)

    def bind(self, trainer):
        """
        Binds the Training Observer to a given trainer

        Parameters
        ----------
        Trainer
            a trainer object

        Returns
        -------
        Training_Observer
            the training observer itself
        """

        trainer.register_training_observer(self)
        return self

    def unbind(self, trainer):
        """
        Unbinds the Training Observer to a given trainer

        Parameters
        ----------
        Trainer
            a trainer object

        Returns
        -------
        Training_Observer
            the training observer itself
        """

        trainer.unregister_training_observer(self)
        return self

    def stateFcn(self, model, input, output, loss, epoch, iteration, t):
        """
        Receives the trainer state and executes the routine functions

        Parameters
        ----------
        model : torch.nn.Module
            the training model
        input : object
            the architecture input
        output : object
            the architecture output
        loss : dict
            a dictionary containing the losses names and values
        epoch : list
            a list containing the starting epoch, the current epoch and the max epoch
        iteration : list
            a list containing the current iteration and the max iteration within an epoch
        t : float
            the execution time of last iteration

        Returns
        -------
        None
        """

        e = epoch
        i = iteration
        g = (e[0]*i[1]+i[0], e[1]*i[1])
        self.iterationFcn(model, input, output, loss, epoch, iteration, t)
        if (g[0] % i[1]) == 0:
            self.epochFcn(model, input, output, loss, epoch, iteration, t)
        if g[0] == (g[1] - 1):
            self.endFcn(model, input, output, loss, epoch, iteration, t)
        return

    def iterationFcn(self, model, input, output, loss, epoch, iteration, t):
        """
        Receives the trainer state at the end of each iteration

        Parameters
        ----------
        model : torch.nn.Module
            the training model
        input : object
            the architecture input
        output : object
            the architecture output
        loss : dict
            a dictionary containing the losses names and values
        epoch : list
            a list containing the starting epoch, the current epoch and the max epoch
        iteration : list
            a list containing the current iteration and the max iteration within an epoch
        t : float
            the execution time of last iteration

        Returns
        -------
        None
        """

        return

    def epochFcn(self, model, input, output, loss, epoch, iteration, t):
        """
        Receives the trainer state at end of each epoch

        Parameters
        ----------
        model : torch.nn.Module
            the training model
        input : object
            the architecture input
        output : object
            the architecture output
        loss : dict
            a dictionary containing the losses names and values
        epoch : list
            a list containing the starting epoch, the current epoch and the max epoch
        iteration : list
            a list containing the current iteration and the max iteration within an epoch
        t : float
            the execution time of last iteration

        Returns
        -------
        None
        """

        return

    def endFcn(self, model, input, output, loss, epoch, iteration, t):
        """
        Receives the trainer state at end of the training

        Parameters
        ----------
        model : torch.nn.Module
            the training model
        input : object
            the architecture input
        output : object
            the architecture output
        loss : dict
            a dictionary containing the losses names and values
        epoch : list
            a list containing the starting epoch, the current epoch and the max epoch
        iteration : list
            a list containing the current iteration and the max iteration within an epoch
        t : float
            the execution time of last iteration

        Returns
        -------
        None
        """

        return
