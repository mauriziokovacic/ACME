from ..utility.islist import *
from .stop_criterion  import *


class TrainingLoop(object):
    """
    A class representing the training loop
    """

    def __init__(self, trainers, criteria=(FrozenModelCriterion(), NaNLossCriterion())):
        self.trainers = trainers
        self.criteria = criteria

    def insert(self, trainer):
        self.trainers += [trainer]
        return self

    def remove(self, trainer):
        self.trainer.remove(trainer)
        return self

    def train(self, dataset, epochs=None, num_acc=None, checkpoint=True, path=None, verbose=False):
        for trainer in self.trainers:
            if not trainer.is_ready():
                raise RuntimeError('Trainer {} is not ready.'.format(trainer.name))
            trainer.model.train()
            trainer.model.zero_grad()

        if epochs is None:
            epochs = len(dataset)

        n = len(dataset)
        if (num_acc is None) or (num_acc <= 0):
            num_acc = 1
        if num_acc > n:
            num_acc = n

        for epoch in range(0, epochs):
            for trainer in self.trainers:
                trainer.optimizer.zero_grad()
                for i, data in enumerate(dataset):
                    result = trainer.train(
                        data,
                        epochs=(epoch, epochs),
                        iters=(i, n),
                        num_acc=num_acc,
                        verbose=verbose,
                    )
                    if self.criteria is not None:
                        for c in self.criteria:
                            if c(result):
                                raise RuntimeError('Stop criterion met')
                if checkpoint:
                    trainer.save_checkpoint(path=path)
        return self

    def __setattr__(self, key, value):
        if key is 'criteria':
            if (value is not None) and (not islist(value)):
                value = [value]
        self.__dict__[key] = value
