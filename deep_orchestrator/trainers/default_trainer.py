from deep_orchestrator.base.callback import BaseCallback
from deep_orchestrator.base.trainer import BaseTrainer
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import torch


class DefaultTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_epochs = self.params.get("max_epochs", 1000)

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

    def prepare_trainer(self, logger, callback):
        self.trainer = pl.Trainer(
            logger=logger,
            max_epochs=self.max_epochs,
            callbacks=[callback]
        )

    def train(self, module, checkpoint_path: str = None):

        assert self.train_dataloader is not None, "Data loader not initialized. Call prepare_data() first."
        assert self.val_dataloader is not None, "Data loader not initialized. Call prepare_data() first."
        assert self.trainer is not None, "Trainer not initialized. Call prepare_trainer() first."

        if checkpoint_path is None:
            self.trainer.fit(module, self.train_dataloader, self.val_dataloader)
        else:
            self.trainer.fit(module, self.train_dataloader, self.val_dataloader, ckpt_path=checkpoint_path)
