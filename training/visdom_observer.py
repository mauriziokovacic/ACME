from ..visdom.VisdomScene               import *
from ..visdom.pie.TrainIterPiePlot      import *
from ..visdom.line.LossPlot             import *
from ..visdom.line.GradientFlowLinePlot import *
from .trainer_observer                  import *


class VisdomObserver(TrainerObserver):
    """
    An observer created with a Visdom session

    Automatically plots dataset processing stats, loss and gradient flow

    Attributes
    ----------
    scene : VisdomScene
        the visdom scene where to plot
    """

    def __init__(self, trainer=None, env='main'):
        """
        Parameters
        ----------
        trainer : Trainer (optional)
            the trainer to observe
        env : str (optional)
            visdom environment (default is 'main')
        """

        super(VisdomObserver, self).__init__(trainer=trainer)
        self.scene = VisdomScene(env=env)
        self.scene.insert_plot(name='proc', cls=TrainIterPiePlot)
        self.scene.insert_plot(name='loss', cls=LossPlot)
        self.scene.insert_plot(name='grad', cls=GradientFlowLinePlot)

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

        self.scene.update_plot('proc', epoch, iteration, t)
        self.scene.update_plot('loss', loss)
        self.scene.update_plot('grad', model)
        super(VisdomObserver, self).stateFcn(name, model, input, output, loss, epoch, iteration, t)
