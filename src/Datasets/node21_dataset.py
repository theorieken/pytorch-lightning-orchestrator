import os
import random
import torch
import pandas as pd
import numpy as np
import SimpleITK as sitk
from PIL import Image

from deep_orchestrator.base.dataset import BaseDataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, Grayscale, ToPILImage


class Node21Dataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dir = self.params.get("path")
        self.fraction = self.params.get("fraction", 1.0)
        self.image_size = self.params.get("image_size", (512, 512))
        self.image_size = max(self.params.get("image_size", (512, 512)))

        # Define the preprocessing pipeline
        self.preprocess = Compose([

            # Resize to a fixed size
            Resize(self.image_size),

            # Convert PIL image to PyTorch tensor
            ToTensor(),

            # Convert the image back to PIL image
            # ToPILImage(),

            # Normalize to [-1, 1]
            # Normalize(mean=[0.5], std=[0.5])

        ])

        self.image_paths, self.label_data = self.load_paths_and_labels()

    def load_paths_and_labels(self):
        image_dir = os.path.join(self.data_dir, "cxr_images", "proccessed_data", "images")
        label_csv_path = os.path.join(self.data_dir, "cxr_images", "proccessed_data", "metadata.csv")

        label_data = pd.read_csv(label_csv_path)

        # Check if each image exists before adding it to the list
        image_paths = [os.path.join(image_dir, img_name) for img_name in label_data["img_name"] if os.path.exists(os.path.join(image_dir, img_name))]

        # Match label data rows to existing images
        label_data = label_data[label_data["img_name"].isin([os.path.basename(path) for path in image_paths])]

        # Shuffle the data and select a subset if self.fraction < 1
        if self.fraction < 1.0:
            combined = list(zip(image_paths, label_data.values))
            random.shuffle(combined)
            num_samples = int(len(combined) * self.fraction)
            image_paths, label_data_values = zip(*combined[:num_samples])
            label_data = pd.DataFrame(np.array(label_data_values), columns=label_data.columns)

        return list(image_paths), label_data

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

        # Apply the preprocessing pipeline
        image = self.preprocess(image)

        # Get label data for this image
        label_data = self.label_data.iloc[idx]

        original_width, original_height = sitk_image.GetSize()
        width_ratio = self.image_size / original_width
        height_ratio = self.image_size / original_height

        labels = torch.Tensor([
            label_data['label'],
            label_data['height'] * height_ratio,
            label_data['width'] * width_ratio,
            label_data['x'] * width_ratio,
            label_data['y'] * height_ratio,
        ])

        return image, labels

    def __len__(self):
        return len(self.image_paths)
