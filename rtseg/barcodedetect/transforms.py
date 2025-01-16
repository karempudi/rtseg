
import albumentations as A
import cv2
import numpy as np
import torch
from torchvision import transforms

class YoloAugmentations:
    
    def __init__(self, parameters = {
        'to_float': {'max_value': 65535.0},
        'random_brightness_contrast': { 'p': 0.5,},
        'random_gamma': { 'p': 0.5 },
        'sharpen': {'p': 0.5},
        'random_horizontal_flip': { 'p': 0.5 },
        'random_vertical_flip': { 'p': 0.5 },
        'shift_scale_rotate': {'rotate_limit': 5, 'interpolation' : cv2.INTER_NEAREST , 'border_mode': cv2.BORDER_CONSTANT},
        'from_float': {'max_value': 65535.0}
    }, bbox_format='yolo', img_size=(256, 800)):
        self.parameters = parameters
        self.transform = A.Compose([
                A.ToFloat(**parameters['to_float']),
                #A.BBoxSafeRandomCrop(always_apply=True),
                A.RandomBrightnessContrast(**parameters['random_brightness_contrast']),
                A.RandomGamma(**parameters['random_gamma']),
                #A.Sharpen(**parameters['sharpen']),
                A.OneOf([
                    A.RandomSizedBBoxSafeCrop(height=img_size[0], width=img_size[1],
                                              always_apply=True, interpolation=cv2.INTER_LINEAR),
                    A.Resize(height=img_size[0], width=img_size[1],always_apply=True, interpolation=cv2.INTER_LINEAR),
                ], p=1.0),
                A.HorizontalFlip(**parameters['random_horizontal_flip']),
                A.VerticalFlip(**parameters['random_vertical_flip']),
                A.ShiftScaleRotate(**parameters['shift_scale_rotate']),
                A.FromFloat(**parameters['from_float']),
            ], bbox_params=A.BboxParams(format=bbox_format))
        
        self.img_size = img_size
    
    def __call__(self, datapoint):
        # datapoint in a dict with keys = {'image', 'bboxes'}
        bboxes = datapoint['bboxes'][:, 1:].tolist()
        for i, box in enumerate(bboxes):
            box.append(datapoint['bboxes'][i, 0]) # add back the label at the end as albumentations expects it
        transformed = self.transform(image=datapoint['image'], bboxes=bboxes)
        
        
        image = transforms.ToTensor()(transformed['image'][:,:, 0].astype('float32'))
        
        bboxes_t = torch.from_numpy(np.array(transformed['bboxes']))
        #print(bboxes_t)
        n_boxes = bboxes_t.shape[0]
        if n_boxes == 0:
            return self.__call__(datapoint)
        bb_targets = torch.zeros((n_boxes, 6))
        assert n_boxes != 0, "number of bboxes after transformation is zero"
        assert bb_targets.shape[0] == bboxes_t.shape[0], "Failed due to shape mismatch in bbox transforms"
        bb_targets[:, 2:] = bboxes_t[:, :4]
        bb_targets[:, 1] = bboxes_t[:, -1]
        #print(bb_targets)
        return {
            'image': image,
            'bboxes': bb_targets
        }


class YoloTestAugmentations:

    """
    Test time augmentations done on the image before the yolo net sees
    and image, basically doing image resizing and tensorizing

    Args:
        parameters: parameters used by various transformations.

    """

    def __init__(self, parameters = {
        'to_float' : {'max_value': 65535.0},
        'resize': {'height': 256, 'width': 800, 'interpolation': cv2.INTER_LINEAR, 'always_apply': True},
        'from_float': {'max_value': 65535.0}
    }):
        self.parameters = parameters
        self.transform = A.Compose([
            A.ToFloat(**parameters['to_float']),
            A.Resize(**parameters['resize']),
            A.FromFloat(**parameters['from_float'])
        ])
        self.img_size = (self.parameters['resize']['height'], self.parameters['resize']['width'])
    def __call__(self, datapoint):
        # datapoint is a dict with keys = {'image'}
        transformed = self.transform(image=datapoint['image'])
        image = transforms.ToTensor()(transformed['image'][:, :, 0].astype('float32'))
        return {
            'image': image,
        }
