import numpy as np
from PIL import Image

from torchvision.transforms.functional import resize


def synthesize_xray(image, direction: str = "frontal"):

    xray = image

    # TODO: Add code to synthesize a frontal xray from a lateral xray

    return xray


def remove_padding(image):
    # Convert the image to a numpy array
    array = np.array(image)

    # Get a binary mask of the pixels where the value is not zero (or 255 for white padding)
    mask = array > 0  # or array < 255

    # Get the coordinates of the non-zero pixels
    coords = np.argwhere(mask)

    # Find the minimum and maximum coordinates on each axis
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)

    # Crop the image to these coordinates
    cropped_image = Image.fromarray(array[x_min:x_max, y_min:y_max])

    return cropped_image


def resize_image(img, size):
    return resize(img, size, interpolation=Image.NEAREST)


def crop_top(img, output_size):
    width, height = img.size
    new_width, new_height = output_size

    # Adjust these parameters to move the cropped square up
    left = (width - new_width)/2
    top = 0  # Start from the top
    right = (width + new_width)/2
    bottom = new_height  # Bottom is the height of the cropped area

    return img.crop((left, top, right, bottom))
