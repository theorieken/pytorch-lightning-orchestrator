from torch.utils.data import DataLoader
from pytorch_lightning import Trainer as PLTrainer


class BaseTrainer:
    def __init__(self, params):
        self.params = params
        self.train_dataloader = None
        self.val_dataloader = None

    def prepare_data(self, dataset):
        raise NotImplementedError("This method needs to be implemented in the child class")

    def train(self, module, logger):
        raise NotImplementedError("This method needs to be implemented in the child class")
