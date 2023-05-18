from deep_orchestrator.base.logger import BaseLogger
from pytorch_lightning.loggers import CSVLogger


class CsvLogger(CSVLogger, BaseLogger):
    def __init__(self, *args, **kwargs):
        # Extract the conf
        conf = kwargs.pop('conf', {})

        super().__init__(**conf)
        self.params = kwargs

    def log_metrics(self, metrics, step=None):
        super().log_metrics(metrics, step)

    def log_message(self, message, kind="info"):
        print(message)

    def on_step(self, trainer, pl_module, outputs, batch):
        pass

    def on_epoch(self, trainer, pl_module, batch):
        pass
