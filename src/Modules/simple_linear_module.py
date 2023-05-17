import torch
import torch.nn as nn
import torch.nn.functional as F
from deep_orchestrator.base.base_module import BaseModule


class SimpleLinearModule(BaseModule):
    def __init__(self, *args, **kwargs):
        super(SimpleLinearModule, self).__init__(*args, **kwargs)

        # Hyperparameters
        self.learning_rate = self.params["learning_rate"]

        # # Simple linear model
        # self.model = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(512 * 512, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 4)
        # )

        # U-Net model
        self.downsample = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 4, kernel_size=1, stride=1)
        )

    # def forward(self, x):
    #     return self.model(x)

    def forward(self, x):
        # Downsample path
        x1 = self.downsample(x)
        x2 = self.downsample(x1)
        x3 = self.downsample(x2)

        # Upsample path
        x = self.upsample(x3)
        x = torch.cat([x, x3], dim=1)
        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.upsample(x)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # Use mean squared error loss for bounding box regression
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
