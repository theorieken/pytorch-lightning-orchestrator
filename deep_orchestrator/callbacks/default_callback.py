from deep_orchestrator.base.callback import BaseCallback


class DefaultCallback(BaseCallback):

    def __init__(self, *args, **kwargs):
        super(DefaultCallback, self).__init__(*args, **kwargs)
        self.last_batch = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.logger.on_train_batch_end(trainer, pl_module, outputs, batch)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx: int = 0):
        self.last_batch = batch

    def on_validation_epoch_end(self, trainer, pl_module):
        self.logger.on_validation_epoch_end(trainer, pl_module, self.last_batch)
