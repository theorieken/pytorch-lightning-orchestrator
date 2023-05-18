import os
import random
import torch
import pandas as pd
import numpy as np
import SimpleITK as sitk
from PIL import Image

from deep_orchestrator.base.dataset import BaseDataset
from torchvision.transforms.functional import resize
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, Grayscale, ToPILImage, Lambda

from transforms import remove_padding, resize_image, crop_top, synthesize_xray
from ..Modules.nodule_module import BaseNoduleModule


class Node21Dataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dir = self.params.get("path")
        self.fraction = self.params.get("fraction", 1.0)
        self.image_size = self.params.get("image_size", (512, 512))

        self.image_paths, self.label_data = self.load_paths_and_labels()

    def load_paths_and_labels(self):

        # TODO: load the data from the luna16 dataset

        return [], []

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        # Define the preprocessing pipeline
        preprocess = Compose([

            # Synthesize an x-ray based on the luna data
            Lambda(synthesize_xray),

            # Remove padding
            Lambda(remove_padding),

            # Crop the image from the top
            Lambda(lambda img: crop_top(img, (min(img.size), min(img.size)))),

            # Resize the cropped square to the desired size
            Lambda(lambda img: img.resize(self.image_size)),

            # Convert PIL image to PyTorch tensor
            ToTensor(),

        ])

        # Read the .mha file using SimpleITK
        sitk_image = sitk.ReadImage(image_path)

        # Get label data for this image
        label_data = self.label_data.iloc[idx]

        original_width, original_height = sitk_image.GetSize()
        width_ratio = self.image_size[0] / original_width
        height_ratio = self.image_size[1] / original_height

        labels = torch.Tensor([
            label_data['label'],
            label_data['height'] * height_ratio,
            label_data['width'] * width_ratio,
            label_data['x'] * width_ratio,
            label_data['y'] * height_ratio,
        ])

        # Convert the SimpleITK image to a NumPy array
        np_array = sitk.GetArrayFromImage(sitk_image)

        # Convert the NumPy array to a PIL image
        image = Image.fromarray(np_array)

        # Apply the preprocessing pipeline
        image = preprocess(image)

        # Convert output to float
        image = image.to(torch.float32)

        # BaseNoduleModule.visualize(image, labels)

        return image, labels

    def __len__(self):
        return len(self.image_paths)
