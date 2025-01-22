import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as TF
import random
from skimage.exposure import adjust_gamma, rescale_intensity


class changedtoPIL(object):

    def __call__(self, sample):
        phase_img = sample['phase'].astype('int16')
        sample['phase'] = TF.to_pil_image(phase_img)
        if ('mask' in sample) and (sample['mask'] is not None):
            mask_img= sample['mask']
            sample['mask'] = TF.to_pil_image(mask_img)
        if ('channel_mask' in sample) and (sample['channel_mask'] is not None):
            channel_mask_img = sample['channel_mask']
            sample['channel_mask'] = TF.to_pil_image(channel_mask_img)
        
        if ('weights' in sample) and (sample['weights'] is not None):
            weights_img = sample['weights'].astype('float32')
            sample['weights'] = TF.to_pil_image(weights_img)
        
        return sample


class randomCrop(object):

    def __init__(self, output_size):
        if isinstance(output_size, tuple):
            self.output_size = output_size
        elif isinstance(output_size, int):
            self.output_size = (output_size, output_size)
    
    def __call__(self, sample):
        i, j, h, w = transforms.RandomCrop.get_params(sample['phase'], output_size=self.output_size)

        sample['phase'] = TF.crop(sample['phase'], i, j, h, w)
        if ('mask' in sample) and (sample['mask'] is not None):
            sample['mask'] = TF.crop(sample['mask'], i, j, h, w)
        if ('channel_mask' in sample) and (sample['channel_mask'] is not None):
            sample['channel_mask'] = TF.crop(sample['channel_mask'], i, j, h, w)
        if ('weights' in sample) and (sample['weights'] is not None):
            sample['weights'] = TF.crop(sample['weights'], i, j, h, w)
        
        return sample

class randomRotation(object):

    def __init__(self, rotation_angle):
        self.rotation_angle = rotation_angle

    def __call__(self, sample):

        angle = transforms.RandomRotation.get_params(self.rotation_angle)
        sample['phase'] = TF.rotate(sample['phase'], angle)
        if ('mask' in sample) and (sample['mask'] is not None):
            sample['mask'] = TF.rotate(sample['mask'], angle)
        if ('channel_mask' in sample) and (sample['channel_mask'] is not None):
            sample['channel_mask'] = TF.rotate(sample['channel_mask'], angle)
        if ('weights' in sample) and (sample['weights'] is not None):
            sample['weights'] = TF.rotate(sample['weights'], angle)

        return sample




class randomAffine(object):

    def __init__(self, scale, shear):
        self.scale = scale # something like 0.75-1.25
        self.shear = shear # something like [-30, 30, -30, 30]

    def __call__(self, sample):

        angle, translations, scale, shear = transforms.RandomAffine.get_params(degrees=[0, 0], translate=None, 
                scale_ranges=self.scale, shears=self.shear, img_size=sample['phase'].size)

        sample['phase'] = TF.affine(sample['phase'], angle=angle, translate=translations, scale=scale, shear=shear)
        if ('mask' in sample) and (sample['mask'] is not None):
            sample['mask'] = TF.affine(sample['mask'], angle=angle, translate=translations, scale=scale, shear=shear)
        if ('channel_mask' in sample) and (sample['channel_mask'] is not None):
            sample['channel_mask'] = TF.affine(sample['channel_mask'], angle=angle, translate=translations, scale=scale, shear=shear)
        if ('weights' in sample) and (sample['weights'] is not None):
            sample['weights'] = TF.affine(sample['weights'], angle=angle, translate=translations, scale=scale, shear=shear)

        return sample


class randomVerticalFlip(object):

    def __call__(self, sample):

        if random.random() < 0.5:
            sample['phase'] = TF.vflip(sample['phase'])
            if ('mask' in sample) and (sample['mask'] is not None):
                sample['mask'] = TF.vflip(sample['mask'])
            if ('channel_mask' in sample) and (sample['channel_mask'] is not None):
                sample['channel_mask'] = TF.vflip(sample['channel_mask'])
            if ('weights' in sample) and (sample['weights'] is not None):
                sample['weights'] = TF.vflip(sample['weights'])

        return sample

class verticalFlip(object):

    def __call__(self, sample):

        sample['phase'] = TF.vflip(sample['phase'])

        if ('mask' in sample) and (sample['mask'] is not None):
            sample['mask'] = TF.vflip(sample['mask'])
        if ('channel_mask' in sample) and (sample['channel_mask'] is not None):
            sample['channel_mask'] = TF.vflip(sample['channel_mask'])
        if ('weights' in sample) and (sample['weights'] is not None):
            sample['weights'] = TF.vflip(sample['weights'])

        return sample

class randomContrast(object):


    def __init__(self, gamma_range):
        self.gamma_range = gamma_range

    def __call__(self, sample):

        if random.random() < 0.5:
            phase_img = np.array(sample['phase']).astype('float32')
            gamma_factor = random.uniform(self.gamma_range[0], self.gamma_range[1])
            brightness_factor = random.uniform(0.0, 5000.0)
            phase_img += brightness_factor
            phase_adjusted = adjust_gamma(phase_img, gamma=gamma_factor)
            phase_adjusted = rescale_intensity(phase_adjusted, in_range='image', out_range='int16')
            sample['phase'] = TF.to_pil_image(phase_adjusted)
            #sample['phase'] = TF.adjust_brightness(sample['phase'], brightness_factor=brightness_factor)
        return sample



class toTensor(object):


    def __call__(self, sample):

        if not isinstance(sample['phase'], np.ndarray):
            sample['phase'] = np.array(sample['phase']).astype(np.float32)

        sample['phase'] = transforms.ToTensor()(sample['phase'])

        if ('mask' in sample) and (sample['mask'] is not None):
            if not isinstance(sample['mask'], np.ndarray):
                sample['mask'] = np.array(sample['mask']).astype(np.float32)
            sample['mask'] = transforms.ToTensor()(sample['mask'])

        if ('channel_mask' in sample) and (sample['channel_mask'] is not None):
            if not isinstance(sample['channel_mask'], np.ndarray):
                sample['channel_mask'] = np.array(sample['channel_mask']).astype(np.float32)
            sample['channel_mask'] = transforms.ToTensor()(sample['channel_mask'])

        if ('weights' in sample) and (sample['weights'] is not None):
            if not isinstance(sample['weights'], np.ndarray):
                sample['weights'] = np.array(sample['weights']).astype(np.float32)
            sample['weights'] = transforms.ToTensor()(sample['weights'])
        
        return sample

def _normalize(img, lower=0.01, upper=99.99):
    img_copy = img.copy()
    return np.interp(img_copy, (np.percentile(img_copy, lower), np.percentile(img_copy, upper)), (0, 1)).astype('float32')

class normalize(object):

    def __call__(self, sample):

        if not isinstance(sample['phase'], np.ndarray):
            phase_img = np.array(sample['phase']).astype('float32')
        else:
            phase_img = sample['phase']
        #phase_img = _normalize(phase_img)
        #sample['phase'] = (phase_img - np.mean(phase_img))/ np.std(phase_img)
        phase_img /= 65535.0
        sample['phase'] = phase_img

        if ('mask' in sample) and (sample['mask'] is not None):
            if not isinstance(sample['mask'], np.ndarray):
                sample['mask'] = np.array(sample['mask']).astype(np.float32)
            sample['mask'] /= 255.0

        if ('channel_mask' in sample) and (sample['channel_mask'] is not None):
            if not isinstance(sample['channel_mask'], np.ndarray):
                sample['channel_mask'] = np.array(sample['channel_mask']).astype(np.float32)
            sample['channel_mask'] /= 255.0

        if ('weights' in sample) and (sample['weights'] is not None):
            if not isinstance(sample['weights'], np.ndarray):
                sample['weights'] = np.array(sample['weights']).astype(np.float32)
 
        return sample


class UnetTrainTransforms:

    def __init__(self, output_size=(512, 512), rotation=[-30, 30], affine_scale=(0.65, 1.35), gamma_range=(0.8, 1.2),
                       affine_shear=[-30, 30, -30, 30], vflip=True, normalize_phase=True, tensorize=True):
        self.transform = []
        self.transform.append(changedtoPIL())
        self.transform.append(randomCrop(output_size))
        self.transform.append(randomContrast(gamma_range))
        self.transform.append(randomRotation(rotation))
        self.transform.append(randomAffine(affine_scale, affine_shear))
        if vflip:
            self.transform.append(randomVerticalFlip())
        if normalize_phase:
            self.transform.append(normalize())
        if tensorize:
            self.transform.append(toTensor())
        
        self.transform = transforms.Compose(self.transform)

    def __call__(self, sample):
        return self.transform(sample) 

class padTo(object):

    def __init__(self, pad_to=8, pad_value=-0.5):
        self.pad_to = pad_to
        self.pad_value = pad_value
    
    def __call__(self, sample):

        image = sample['phase']
        H, W = image.shape
        
        if H % self.pad_to != 0:
            H_t = ((H // self.pad_to) + 1) * self.pad_to 
        else:
            H_t = H
        if W % self.pad_to != 0:
            W_t = ((W // self.pad_to) + 1) * self.pad_to
        else:
            W_t = W

        image = np.pad(image, pad_width= ((0, H_t - H), (0, W_t - W)),
                             mode='constant', constant_values=self.pad_value)

        sample['phase'] = image
        return sample
        

class unPad(object):

    def __call__(self, image, shape):
        H, W = shape
        return image[..., :H, :W]

class UnetTestTransforms:

    def __init__(self, normalize_phase=True, tensorize=True, vflip=False, pad_to=16):
        self.transform = []
        if normalize_phase:
            self.transform.append(normalize())
        if pad_to > 0:
            self.transform.append(padTo(pad_to))
        
        if tensorize:
            self.transform.append(toTensor())
        if vflip:
            self.transform.append(verticalFlip())
        
        self.transform = transforms.Compose(self.transform)

    def __call__(self, sample):
        return self.transform(sample)