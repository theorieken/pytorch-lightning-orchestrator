import os
import pandas as pd
import numpy as np
import torch
from deep_orchestrator.base.base_dataset import BaseDataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, Grayscale
import SimpleITK as sitk
from PIL import Image


class Node21Dataset(BaseDataset):
    def __init__(self, params):
        super().__init__(params)
        self.data_dir = params["path"]
        self.image_size = params.get("image_size", (512, 512))  # New attribute for the target image size

        # Define the preprocessing pipeline
        self.preprocess = Compose([
            Resize(self.image_size),  # Resize the images
            Grayscale(num_output_channels=1),  # Convert to grayscale
            ToTensor(),  # Convert PIL image to PyTorch tensor
            Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
        ])

        self.image_paths, self.label_data = self.load_paths_and_labels()

        self.image_paths, self.label_data = self.load_paths_and_labels()

    def load_paths_and_labels(self):
        image_dir = os.path.join(self.data_dir, "cxr_images", "original_data", "images")
        label_csv_path = os.path.join(self.data_dir, "cxr_images", "original_data", "metadata.csv")

        label_data = pd.read_csv(label_csv_path)
        image_paths = [os.path.join(image_dir, img_name) for img_name in label_data["img_name"]]

        return image_paths, label_data

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        # Read the .mha file using SimpleITK
        sitk_image = sitk.ReadImage(image_path)

        # Convert the SimpleITK image to a NumPy array
        np_array = sitk.GetArrayFromImage(sitk_image)

        # Convert the NumPy array to a compatible data type if needed
        np_array = np_array.astype(np.uint8)

        # Convert the NumPy array to a PIL image
        image = Image.fromarray(np_array)

        # Transform the image into RGB
        image = image.convert("RGB")

        # Apply the preprocessing pipeline
        image = self.preprocess(image)

        # Get label data for this image
        label_data = self.label_data.iloc[idx]

        # Create a tensor for the bounding box coordinates
        # Normalized to [0, 1] assuming image size is 512x512
        bbox = torch.tensor([label_data["x"], label_data["y"], label_data["width"], label_data["height"]]) / 512.0

        return image, bbox  # Channel dimension is already added by ToTensor

    def __len__(self):
        return len(self.image_paths)
