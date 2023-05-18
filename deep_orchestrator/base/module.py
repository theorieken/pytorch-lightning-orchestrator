import pytorch_lightning as pl


class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.params = kwargs

    def forward(self, x):
        raise NotImplementedError("This method needs to be implemented in the child class")

    def training_step(self, batch, batch_idx):
        raise NotImplementedError("This method needs to be implemented in the child class")

    def configure_optimizers(self):
        raise NotImplementedError("This method needs to be implemented in the child class")
