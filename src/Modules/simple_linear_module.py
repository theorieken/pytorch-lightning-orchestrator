import torch
import torch.nn as nn
import torch.nn.functional as F
from .nodule_module import BaseNoduleModule
from deep_orchestrator.base.module import BaseModule


class SimpleLinearModule(BaseNoduleModule, BaseModule):
    def __init__(self, *args, **kwargs):
        super(SimpleLinearModule, self).__init__(*args, **kwargs)

        self.train_loss = []
        self.val_loss = []

        self.last_batch = None

        # Hyperparameters
        self.learning_rate = self.params["learning_rate"]
        self.simple_version = self.params.get('simple', False)

        if self.simple_version:

            # Simple linear model
            self.model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512 * 512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 5)
            )

        else:

            # U-Net model
            self.downsample1 = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

            self.downsample2 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

            self.downsample3 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

            self.upsample1 = nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 64, kernel_size=1, stride=1)
            )

            self.upsample2 = nn.Sequential(
                nn.ConvTranspose2d(64 + 128, 64, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 4, kernel_size=1, stride=1)
            )

            self.flatten = nn.Flatten()
            self.regression = nn.Sequential(
                nn.Linear(64 * 128 * 128, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 5)
            )

    def forward(self, x):

        if self.simple_version:
            return self.model(x)
        else:

            # Downsample path
            x1 = self.downsample1(x)
            x2 = self.downsample2(x1)
            x3 = self.downsample3(x2)

            # Upsample path
            x = self.upsample1(x3)
            x = torch.cat([x, x2], dim=1)
            x = self.upsample2(x)

            # Flatten and regression stage
            x = self.flatten(x)
            bbox = self.regression(x)

            return bbox

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # Use mean squared error loss for bounding box regression
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.train_loss.append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.val_loss.append(loss)
        self.last_batch = batch

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
