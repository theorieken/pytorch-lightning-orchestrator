class BaseLogger:

    def __init__(self, *args, **kwargs):
        self.params = kwargs

    def log_metrics(self, metrics, step=None):
        raise NotImplementedError("log_metrics method not implemented")

    def log_message(self, message, kind="info"):
        raise NotImplementedError("log_message method not implemented")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch):
        current_epoch = trainer.current_epoch
        global_step = trainer.global_step
        metrics = {**{"epoch": current_epoch, "step": global_step}, **trainer.callback_metrics}
        self.log_metrics(metrics)

        # call the on step method
        self.on_step(trainer, pl_module, outputs, batch)

    def on_validation_epoch_end(self, trainer, pl_module, batch):
        current_epoch = trainer.current_epoch
        global_step = trainer.global_step
        metrics = {**{"epoch": current_epoch, "step": global_step}, **trainer.callback_metrics}
        self.log_metrics(metrics)

        # call the on epoch method
        self.on_epoch(trainer, pl_module, batch)

    def on_step(self, trainer, pl_module, outputs, batch):
        pass

    def on_epoch(self, trainer, pl_module, batch):
        pass
