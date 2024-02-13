
import numpy as np
import numpy.random as random

from math import ceil


def v_flip(image):
    return np.ascontiguousarray(image[:, ::-1, ...])


def h_flip(image):
    return np.ascontiguousarray(image[:, :, ::-1, ...])


def rot90(image, factor):
    return np.ascontiguousarray(np.rot90(image, factor, axes = [1, 2]))


def random_crop_coords(height, width, crop_height, crop_width):
    h_start = random.random()
    w_start = random.random()

    if (height < crop_height) or (width < crop_width):
        raise ValueError(
            f"Crop size ({crop_height}, {crop_width}) larger than the "
            f"image size ({height}, {width})."
        )

    y1 = int((height - crop_height + 1) * h_start)
    y2 = y1 + crop_height

    x1 = int((width - crop_width + 1) * w_start)
    x2 = x1 + crop_width

    return y1, y2, x1, x2


class Compose:
    """
    Iterate through all the transform in order given and return
    the final transformed image, mask and vf
    Args:
        layers : a list of other intialized transformations

    """
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, image, mask, vf = None):
        for layer in self.layers:
            image, mask, vf = layer(image, mask, vf)
        
        return image, mask, vf

class RandomCrop:
    def __init__(self, crop_h, crop_w):
        self.crop_h = crop_h
        self.crop_w = crop_w

    def __call__(self, image, mask, vf = None):
        """
        None

        Args:
            image (List[np.ndarray]): List of numpy arrays, each
                of shape (C, H, W). 

        Returns: 
            List[np.ndarray]: List of numpy arrays, each of shape
                (C, crop_h, crop_w).

        """
        _, height, width = image.shape

        if width < self.crop_w:
            pad = ceil((self.crop_w - width) / 2)
            padding = ((0, 0), (0, 0), (pad, pad))

            image = np.pad(image, padding)
            mask = np.pad(mask, padding)

            if vf is not None:
                vf = np.pad(vf, padding)

        if height < self.crop_h:
            pad = ceil((self.crop_h - height) / 2)
            padding = ((0, 0), (pad, pad), (0, 0))

            image = np.pad(image, padding)
            mask = np.pad(mask, padding)

            if vf is not None:
                vf = np.pad(vf, padding)

        _, height, width = image.shape
        y1, y2, x1, x2 = random_crop_coords(
            height,
            width,
            self.crop_h,
            self.crop_w
        )

        image = image[:, y1 : y2, x1 : x2]
        mask = mask[:, y1 : y2, x1 : x2]

        if vf is not None:
            vf = vf[:, y1 : y2, x1 : x2]

        return image, mask, vf

class ToFloat:
    
    def __init__(self, max_value):
        self.max_value = max_value

    def __call__(self, image, mask, vf = None):
        image = image.astype(np.float32) / self.max_value
        mask = mask.astype(np.float32)

        if vf is not None: 
            vf = vf.astype(np.float32)

        return image, mask, vf


class HorizontalFlip:
    def __init__(self, p):
        self.p = p

    def __call__(self, image, mask, vf = None):
        if random.random() < self.p:
            image = h_flip(image)
            mask  = h_flip(mask)

            if vf is not None:
                vf = h_flip(vf)

                # Must change the order of VF if doing
                # horizontal flip.
                vf[0] = -vf[0]

        return image, mask, vf

class VerticalFlip:
    def __init__(self, p):
        self.p = p

    def __call__(self, image, mask, vf = None):
        if random.random() < self.p:
            image = v_flip(image)
            mask  = v_flip(mask)

            if vf is not None:
                vf = v_flip(vf)

                # Must change the order of VF if doing
                # vertical flip.
                vf[1] = -vf[1]


        return image, mask, vf

class AddDimension:

    def __init__(self):
        pass

    def __call__(self, image, mask, vf = None):

        image = image[None]
        mask = mask[None]

        return image, mask, vf


train_transform = Compose([
    AddDimension(),
    RandomCrop(320, 320),
    VerticalFlip(p = 0.25),
    HorizontalFlip(p = 0.25),
    ToFloat(255),
])

eval_transform = Compose([
    AddDimension(),
    ToFloat(255)
])

transforms = {
    "train": train_transform,
    "eval": eval_transform,
    None: None
}