from deep_orchestrator.base.base_logger import BaseLogger
import wandb


class BoundingBoxLogger(BaseLogger):

    def tensor_to_wandb_image(self, tensor, bbox, bbox_predict):

        # Convert tensor to numpy array and s queeze out the channel dimension
        image = tensor.squeeze(0).numpy()

        # Only use first channel
        bbox_predict = bbox_predict[0]

        # Normalize the image to 0-255 if it's not already in that range
        image = ((image - image.min()) * (1 / (image.max() - image.min()) * 255)).astype('uint8')

        # Define the box labels
        class_labels = {0: "nodule", 1: "prediction"}

        # Create a wandb Image
        wandb_image = wandb.Image(
            image,
            boxes={
                "predictions": {
                    "box_data": [
                        {
                            # Position of the box
                            "position": {
                                "middle": [int(int(bbox[3]) + int(bbox[2]) / 2), int(int(bbox[4]) + int(bbox[1]) / 2)],
                                "width": int(bbox[2]),
                                "height": int(bbox[1]),
                            },
                            # optionally caption each box with its class and score
                            "domain": "pixel",
                            "class_id": 0,
                            "box_caption": "nodule",
                            "scores": {"acc": 0.5, "loss": 0.7},
                        },
                        {
                            # Position of the box
                            "position": {
                                "middle": [int(int(bbox_predict[3]) + int(bbox_predict[2]) / 2), int(int(bbox_predict[4]) + int(bbox_predict[1]) / 2)],
                                "width": int(bbox_predict[2]),
                                "height": int(bbox_predict[1]),
                            },
                            # optionally caption each box with its class and score
                            "domain": "pixel",
                            "class_id": 1,
                            "box_caption": "prediction",
                            "scores": {"acc": 0.5, "loss": 0.7},
                        }
                    ],
                    "class_labels": class_labels
                }
            }
        )
        return wandb_image

    def on_epoch(self, trainer, pl_module):
        a = 0
