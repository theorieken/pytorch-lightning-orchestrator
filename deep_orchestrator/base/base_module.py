import pytorch_lightning as pl


class BaseModule(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass
