from pytorch_lightning.loggers import WandbLogger
import logging


class BaseLogger(WandbLogger):
    def __init__(self, params, wandb, job_dir):
        super().__init__(**wandb)
        self.params = params
        self.job_name = params.get('job_name', 'unnamed_job')
        self.job_dir = job_dir

        # Create logger
        self.logger = logging.getLogger(self.job_name)
        self.logger.setLevel(logging.INFO)

        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)

        # Add the handlers to the logger
        self.logger.addHandler(ch)

    def log_message(self, message: str):
        self.logger.info(message)
        with open(f"{self.job_dir}/log.txt", "a") as log_file:
            log_file.write(message + "\n")

    def log_metrics(self, metrics, step=None):
        super().log_metrics(metrics, step)

    def on_step(self, trainer, pl_module):
        # Override this method to do something after each step
        pass

    def on_epoch(self, trainer, pl_module):
        # Override this method to do something after each epoch
        pass
