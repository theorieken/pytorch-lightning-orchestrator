import os
import pandas as pd
import torch
import SimpleITK as sitk
from deep_orchestrator.base.base_dataset import BaseDataset


class Node21Dataset(BaseDataset):
    def __init__(self, params):
        super().__init__(params)
        self.data_dir = params["path"]
        self.split = params["split"]
        self.fraction = params.get("fraction", 1.0)

        self.image_paths, self.label_data = self.load_paths_and_labels()

    def load_paths_and_labels(self):
        image_dir = os.path.join(self.data_dir, "cxr_images", "original_data", "images")
        label_csv_path = os.path.join(self.data_dir, "cxr_images", "original_data", "metadata.csv")

        label_data = pd.read_csv(label_csv_path)
        image_paths = [os.path.join(image_dir, img_name) for img_name in label_data["img_name"]]

        return image_paths, label_data

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        # Convert SimpleITK image to PyTorch tensor
        image = torch.tensor(sitk.GetArrayFromImage(sitk.ReadImage(image_path)), dtype=torch.float32)

        # Normalize image to [0, 1]
        image = image / 255.0

        # Get label data for this image
        label_data = self.label_data.iloc[idx]

        # Create a tensor for the bounding box coordinates
        # Normalized to [0, 1] assuming image size is 512x512
        bbox = torch.tensor([label_data["x"], label_data["y"], label_data["width"], label_data["height"]]) / 512.0

        return image.unsqueeze(0), bbox  # Add channel dimension to image

    def __len__(self):
        return len(self.image_paths)
