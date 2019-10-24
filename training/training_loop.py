from ..utility.isnan import *


class TrainingLoop(object):
    """
    A class representing the training loop
    """

    def __init__(self, trainers):
        self.trainers = trainers

    def insert(self, trainer):
        self.trainers += [trainer]
        return self

    def remove(self, trainer):
        self.trainer.remove(trainer)
        return self

    def train(self, dataset, epochs=None, num_acc=None, checkpoint=True, path=None, verbose=False):
        for trainer in self.trainers:
            trainer.model.train()
            trainer.model.zero_grad()

        if epochs is None:
            epochs = len(dataset)

        for epoch in range(0, epochs):
            for trainer in self.trainers:
                trainer.optimizer.zero_grad()
                for i, input in enumerate(dataset):
                    result = trainer.train(
                        input,
                        epochs=(epoch, epochs),
                        iters=(i, len(dataset)),
                        num_acc=num_acc,
                        path=path,
                        verbose=verbose)
                    if not result[0]:
                        raise RuntimeError('Something went wrong with Trainer {}'.format(trainer.name))
                    else:
                        if isnan(result[1]):
                            raise RuntimeError('Trainer {} became NaN'.format(trainer.name))
                if checkpoint:
                    trainer.save_checkpoint(path=path)
        return self
