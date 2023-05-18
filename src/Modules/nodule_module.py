from deep_orchestrator.base.logger import BaseLogger
import wandb
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import torch
from PIL import Image


class BaseNoduleModule:

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_batch = None

    @staticmethod
    def visualize(image, bbox):

        # If the image is a tensor, convert it to a numpy array and move channels to the last dimension
        if torch.is_tensor(image):

            image = image.cpu()
            bbox = bbox.cpu()

            image = image.numpy()
            if image.shape[0] == 1:
                # Grayscale image, squeeze out the channel dimension
                image = np.squeeze(image, axis=0)
            else:
                # Color image, move channels to the last dimension
                image = np.transpose(image, (1, 2, 0))

        # Create figure and axes
        fig, ax = plt.subplots(1)

        # Display the image
        ax.imshow(image, cmap='gray' if image.ndim == 2 else None)

        # Add bounding box
        rect = patches.Rectangle((bbox[3], bbox[4]), bbox[2], bbox[1], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        # Show the figure
        plt.show()

    @staticmethod
    def get_wandb_image(tensor, bbox, bbox_predict):

        # Move the tensor to the cpu first
        tensor = tensor.cpu()

        # Convert tensor to numpy array and squeeze out the channel dimension
        image = tensor.squeeze(0).numpy()

        # Normalize the image to 0-255 if it's not already in that range
        image_normed = ((image - image.min()) * (1 / (image.max() - image.min()) * 255)).astype('uint8')

        # Define the box labels
        class_labels = {0: "prediction", 1: "ground truth"}

        # Initialize the box data
        predictions = []

        # Check if nodule is present
        if bbox[0] > 0.5:
            predictions.append({
                # Position of the box
                "position": {
                    "middle": [int(int(bbox[3]) + int(bbox[2]) / 2), int(int(bbox[4]) + int(bbox[1]) / 2)],
                    "width": int(bbox[2]),
                    "height": int(bbox[1]),
                },
                # optionally caption each box with its class and score
                "domain": "pixel",
                "class_id": 0,
                "box_caption": class_labels[0],
                "scores": {"acc": 0.5, "loss": 0.7},
            })

        if bbox_predict[0] > 0.5:
            predictions.append({
                # Position of the box
                "position": {
                    "middle": [int(int(bbox_predict[3]) + int(bbox_predict[2]) / 2), int(int(bbox_predict[4]) + int(bbox_predict[1]) / 2)],
                    "width": int(bbox_predict[2]),
                    "height": int(bbox_predict[1]),
                },
                # optionally caption each box with its class and score
                "domain": "pixel",
                "class_id": 1,
                "box_caption": class_labels[1],
                "scores": {"acc": 0.5, "loss": 0.7},
            })

        # Create a wandb Image
        wandb_image = wandb.Image(
            image_normed,
            boxes={
                "predictions": {
                    "box_data": predictions,
                    "class_labels": class_labels
                }
            }
        )

        return wandb_image

    def on_validation_end(self):

        x, y = self.last_batch
        y_hat = self(x)

        examples = []
        for i in range(len(x)):
            # BaseNoduleModule.visualize(x[i], y[i])
            examples.append(BaseNoduleModule.get_wandb_image(x[i], y_hat[i], y[i]))

        # this only works for wandb logger
        if hasattr(self.trainer.logger, 'log_image'):
            # TODO: rebuild to use logger
            wandb.log({"Prediction examples": examples})
