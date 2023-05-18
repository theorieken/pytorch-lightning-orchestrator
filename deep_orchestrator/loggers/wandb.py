from deep_orchestrator.base.logger import BaseLogger
from pytorch_lightning.loggers import WandbLogger
import logging


class WeightBiasesLogger(WandbLogger, BaseLogger):
    def __init__(self, *args, **kwargs):

        # Extract the conf
        conf = kwargs.pop('conf', {})

        super().__init__(**conf)
        self.params = kwargs
        self.job_name = self.params.get('job_name', 'unnamed_job')

        # Create logger
        self.logger = logging.getLogger(self.job_name)
        self.logger.setLevel(logging.INFO)

        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Create formatter and add it to the handlers TODO: add some cool color encoding
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)

        # Add the handlers to the logger
        self.logger.addHandler(ch)

    def log_message(self, message: str, kind: str = "info"):
        if kind == "info":
            self.logger.info(message)
        elif kind == "warning":
            self.logger.warning(message)
        elif kind == "error":
            self.logger.error(message)
        elif kind == "critical":
            self.logger.critical(message)
        else:
            raise ValueError(f"Unknown message kind '{kind}'")

        with open(f"{self.save_dir}/log.txt", "a") as log_file:
            log_file.write("\n" + message + "\n")

    def log_metrics(self, metrics, step=None):
        super().log_metrics(metrics, step)

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
