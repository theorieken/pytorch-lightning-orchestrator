from pytorch_lightning.loggers import WandbLogger
import logging


class BaseLogger(WandbLogger):
    def __init__(self, params, wandb):
        super().__init__(**wandb)
        self.params = params
        self.job_name = params.get('job_name', 'unnamed_job')

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
        with open(f"{self.save_dir}/log.txt", "a") as log_file:
            log_file.write(message + "\n")

    def log_metrics(self, metrics, step=None):
        super().log_metrics(metrics, step)

    def on_step(self, trainer, pl_module):
        # You can access the logged metrics in the trainer.logged_metrics dict
        train_loss = trainer.logged_metrics.get('train_loss')
        if train_loss is not None:
            self.log_metrics({'train_loss': train_loss}, step=trainer.global_step)

    def on_epoch(self, trainer, pl_module):
        # You can access the logged metrics in the trainer.logged_metrics dict
        val_loss = trainer.logged_metrics.get('val_loss')
        if val_loss is not None:
            self.log_metrics({'val_loss': val_loss}, step=trainer.current_epoch)
