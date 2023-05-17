from deep_orchestrator.base.base_trainer import BaseTrainer
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import torch


class DefaultTrainer(BaseTrainer):
    def __init__(self, params):
        super().__init__(params)

    def prepare_data(self, dataset):
        split_ratio = self.params.get("split_ratio", 0.8)
        batch_size = self.params.get("batch_size", 4)
        shuffle = self.params.get("shuffle", True)
        num_workers = self.params.get("num_workers", 4)
        pin_memory = self.params.get("pin_memory", True)

        # Determine split threshold and perform random split of the passed data set
        split_value = int(split_ratio * len(dataset))
        train_dataset, val_dataset = random_split(dataset, [split_value, len(dataset) - split_value], generator=torch.Generator().manual_seed(42))

        # Initialize data loaders for both parts of the split data set
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
        self.val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

    def train(self, module, logger):
        trainer = pl.Trainer(logger=logger)
        trainer.fit(module, self.train_dataloader, self.val_dataloader)
