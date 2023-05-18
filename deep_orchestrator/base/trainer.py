from torch.utils.data import DataLoader
from pytorch_lightning import Trainer as PLTrainer


class BaseTrainer:
    def __init__(self, *args, **kwargs):
        self.params = kwargs
        self.train_dataloader = None
        self.val_dataloader = None
        self.trainer = None

    def prepare_data(self, dataset):
        raise NotImplementedError("This method needs to be implemented in the child class")

    def prepare_trainer(self, logger, callback):
        raise NotImplementedError("This method needs to be implemented in the child class")

    def train(self, module, checkpoint_path: str = None):
        raise NotImplementedError("This method needs to be implemented in the child class")
