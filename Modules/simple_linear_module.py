import torch
import torch.nn as nn
import torch.nn.functional as F
from deep_orchestrator.base.base_module import BaseModule


class SimpleLinearModule(BaseModule):
    def __init__(self, params):
        super(SimpleLinearModule, self).__init__(params)

        # Hyperparameters
        self.learning_rate = params["learning_rate"]

        # Simple linear model
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * 1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # Use mean squared error loss for bounding box regression
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)