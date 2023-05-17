from deep_orchestrator.base.base_logger import BaseLogger


class DefaultLogger(BaseLogger):
    def on_step(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        global_step = trainer.global_step
        train_loss = pl_module.training_loss
        self.log_metrics({"epoch": current_epoch, "step": global_step, "train_loss": train_loss})

    def on_epoch(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        train_loss = pl_module.training_loss
        val_loss = pl_module.validation_loss
        self.log_metrics({"epoch": current_epoch, "train_loss": train_loss, "val_loss": val_loss})
