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
    __stateFcn(input,output,loss,epoch,iteration,t)
        Receives the trainer state and executes the routine functions
    iterationFcn(input,output,loss,epoch,iteration,t)
        Receives the trainer state at the end of each iteration
    epochFcn(input,output,loss,epoch,iteration,t)
        Receives the trainer state at the end of each epoch
    endFcn(input,output,loss,epoch,iteration,t)
        Receives the trainer state at the end of the training
    """

    def __init__(self,trainer=None):
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



    def bind(self,trainer):
        """
        Binds the Training Observer to a given trainer

        Parameters
        ----------
        Trainer
            a trainer object
        """

        trainer.stateFcn = self.__stateFcn



    def __stateFcn(self,input,output,loss,epoch,iteration,t):
        """
        Receives the trainer state and executes the routine functions

        Parameters
        ----------
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

        e = epoch[1:3]
        i = iteration
        g = (e[0]*i[1]+i[0],e[1]*i[1])
        if not e[0]==self.epoch:
            self.epoch = e[0]
            self.epochFcn(input,output,loss,epoch,iteration,t)
        if not i[0]==self.iter:
            self.iter = i[0]
            self.iterationFcn(input,output,loss,epoch,iteration,t)
        if g[0]==g[1]-1:
            self.endFcn(input,output,loss,epoch,iteration,t)
        return



    def iterationFcn(self,input,output,loss,epoch,iteration,t):
        """
        Receives the trainer state at the end of each iteration

        Parameters
        ----------
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



    def epochFcn(self,input,output,loss,epoch,iteration,t):
        """
        Receives the trainer state at end of each epoch

        Parameters
        ----------
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



    def endFcn(self,input,output,loss,epoch,iteration,t):
        """
        Receives the trainer state at end of the training

        Parameters
        ----------
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
