
import numpy as np
#import numpy.random as random
from torchvision import transforms # type: ignore
import torchvision.transforms.functional as TF # type: ignore
import random
#from math import ceil
from skimage.exposure import adjust_gamma, rescale_intensity
from rtseg.cellseg.numerics.sdf_vf import sdf_vector_field
from skimage.measure import label
import torch

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

    def __call__(self, image, mask):
        inputs = (image, mask)
        for layer in self.layers:
            #print(layer)
            inputs = layer(*inputs)
        return inputs

class RandomCrop:

    def __init__(self, output_size):
        if isinstance(output_size, tuple):
            self.output_size = output_size
        elif isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        
    def __call__(self, image, mask, vf = None):

        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=self.output_size)
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        if vf is None:
            return image, mask

        return image, mask, vf

class RandomRotation:

    def __init__(self, rotation_angle):
        self.rotation_angle = rotation_angle
    
    def __call__(self, image, mask, vf = None):
        angle = transforms.RandomRotation.get_params((-self.rotation_angle, self.rotation_angle))
        #print(angle)

        image = TF.rotate(image, angle)
        mask = TF.rotate(mask, angle)

        if vf is None:
            return image, mask
        
        return image, mask, vf

class RandomBrightness:

    def __init__(self, gamma_range=(0.7, 1.4), brightness_add_range=(-2500.0, 5000.0)):
        self.gamma_range = gamma_range
        self.brightness_add_range = brightness_add_range

    def __call__(self, image, mask, vf = None):
        gamma = random.uniform(*self.gamma_range)
        brightness = random.uniform(*self.brightness_add_range)

        image = adjust_gamma(image, gamma = gamma)
        image += brightness
        image = rescale_intensity(image, in_range='image', out_range='uint16')

        if vf is None:
            return image, mask

        return image, mask, vf


class ToFloat:
    
    def __init__(self, max_value):
        self.max_value = max_value

    def __call__(self, image, mask, vf = None):
        image = image / self.max_value

        if vf is None: 
            return image, mask
        else:
            return image, mask, vf



class HorizontalFlip:
    def __init__(self, p):
        self.p = p

    def __call__(self, image, mask, vf = None):
        #if random.random() < self.p:
        #    image = h_flip(image)
        #    mask  = h_flip(mask)
#
#            if vf is not None:
#                vf = h_flip(vf)
#
#                # Must change the order of VF if doing
#                # horizontal flip.
#                vf[0] = -vf[0]

        if random.random() < self.p:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if vf is None:
            return image, mask

        return image, mask, vf

class VerticalFlip:
    def __init__(self, p):
        self.p = p

    def __call__(self, image, mask, vf = None):
        #if random.random() < self.p:
        #    image = v_flip(image) # always calculate vfs in the end 
        #    mask  = v_flip(mask)
#
#            if vf is not None:
#                vf = v_flip(vf)
#
#                # Must change the order of VF if doing
#                # vertical flip.
#                vf[1] = -vf[1]
        if random.random() < self.p:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        if vf is None:
            return image, mask

        return image, mask, vf

class AddDimension:

    def __init__(self):
        pass

    def __call__(self, image, mask, vf = None):

        image = image[None]
        mask = mask[None]

        if vf is None:
            return image, mask

        return image, mask, vf

class RandomAffine:

    def __init__(self, scale, shear):
        self.scale = scale
        self.shear = shear
    
    def __call__(self, image, mask, vf = None):

        angle, translations, scale, shear = transforms.RandomAffine.get_params(degrees=[0, 0], translate=None, 
        scale_ranges=self.scale, shears=self.shear, img_size=image.size)

        image = TF.affine(image, angle=angle, translate=translations, scale=scale, shear=shear)
        mask = TF.affine(mask, angle=angle, translate=translations, scale=scale, shear=shear)

        if vf is None:
            return image, mask
    
        return image, mask, vf

class AddVectorField:

    def __init__(self, kernel_size):
        self.kernel_size = kernel_size
    
    def __call__(self, image, mask, vf = None):
        
        image = transforms.ToTensor()(image)
        mask = np.array(mask).astype('float32')
        mask = label(mask)

        vf = sdf_vector_field(torch.tensor(mask)[None, :], self.kernel_size)
        mask = torch.tensor(mask)[None, :]

        return image, mask, vf

class changedToPIL:

    def __call__(self, image, mask, vf = None):
        
        image = image.astype('int16')
        image = TF.to_pil_image(image)
        mask = TF.to_pil_image(mask)
        if vf is None:
            return image, mask
        
        return image, mask, vf


train_transform = Compose([
    changedToPIL(),
    RandomCrop(320),
    RandomRotation(20.0),
    RandomAffine(scale=(0.75, 1.25), shear=(-30, 30, -30, 30)),
    VerticalFlip(p = 0.25),
    HorizontalFlip(p = 0.25),
    #RandomBrightness(gamma_range=(0.7, 1.4), brightness_add_range=(-2500.0, 5000.0)),
    AddVectorField(kernel_size=11), # always calculate vfs in the end 
    ToFloat(65535),
])

eval_transform = Compose([
    AddDimension(),
    #RandomBrightness(gamma_range=(0.7, 1.4), brightness_add_range=(-2500.0, 5000.0)),
    ToFloat(65535)
])

all_transforms = {
    "train": train_transform,
    "eval": eval_transform,
    None: None
}