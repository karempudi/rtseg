
from tifffile import imread
import pathlib
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import Dataset
import cv2


class BarcodeDataset(Dataset):
    """
    
    Constructs datapoints that are seen by the network
    
    Arguments:
        data_dir (pathlib.Path or str): path of the data directory 
        img_size (tuple): Default (256, 800) image size seen by the network
        multiscale (boolean): Hanve't figured out this yet TODO
        transform (torchvision transformations): transformations that will be applied to
                each image in the dataset during training (augmentations and tensorizations)
    
    """
    def __init__(self, data_dir, img_size=(256, 800), multiscale=False, transform=None):
        super(BarcodeDataset, self).__init__()
        self.data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir
        assert isinstance(self.data_dir, pathlib.Path) is True, "data directory is not a valid path"
        
        # images are in /images directory
        self.images_dir = self.data_dir / Path('images')
        # labels are in /labels directory
        self.labels_dir = self.data_dir / Path('labels')
        
        self.img_size = img_size
        self.multiscale = multiscale
        self.transform = transform
        
        # build tuples of filenames and labels here
        filenames = self.images_dir.glob('*.tiff')
        filestems = [file.stem for file in filenames]
        self.data_tuples = [(self.images_dir / Path(filestem + '.tiff'),
                             self.labels_dir / Path(filestem + '.txt')) 
                                    for filestem in filestems]

        # not figured out where this will be used exactly
        self.batch_count = 0
        
    def __len__(self):
        return len(self.data_tuples)
    
    def __getitem__(self, idx):
        # get one data point from the dataset that is transformed
        image_filename, bbox_filename = self.data_tuples[idx]
        image = imread(image_filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = np.loadtxt(bbox_filename)
        #for bbox in bboxes:
        #    bbox.append(0)
        
        datapoint = {
            'image': image,
            'bboxes': bboxes,
        }
        
        if self.transform is not None:
            transformed = self.transform(datapoint)
            image = transformed['image']
            bboxes = transformed['bboxes']
        return {
            'image': image,
            'bboxes': bboxes,
            'path': str(image_filename)
        }

    def collate_fn(self, batch):
        self.batch_count += 1 

        # drop invalid images
        batch = [data for data in batch if data is not None]

        # images stack
        images = torch.stack([data['image'] for data in batch])

        # you need to impelement multiscale option here
        # TODO: later. Need to figure out how bboxes need
        # to be transformed for multiscale training
        # for now, we do only one scale

        bb_targets = [data['bboxes'] for data in batch]
        for i, bboxes in enumerate(bb_targets):
            # first col of each bbox in a batch corresponds to image idx in the batch
            bboxes[:, 0] = i
        bb_targets = torch.cat(bb_targets, 0)

        paths = [data['path'] for data in batch]
        
        return paths, images, bb_targets
    
    def plot_datapoint(self, idx):
        datapoint = self.__getitem__(idx)
        image = datapoint['image']
        #print(image.shape)
        bboxes = datapoint['bboxes']
        #print(bboxes)
        if image.ndim == 3:
            image = image[0, :,:,]
        #image = image/65535
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image, cmap='gray')
        height, width = image.shape
        #print(f"Height: {height} -- width: {width}")
        if bboxes is not None:
            for bbox in bboxes:
                w = bbox[4] * width
                h = bbox[5] * height
                length = int((bbox[2] - bbox[4]/2) * width)
                t = int((bbox[3] - bbox[5]/2) * height)
                rect = patches.Rectangle((length, t), w, h, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

        plt.show()


class BarcodeTestDataset(Dataset):
    """
    Constructs dataset that is seen by the network during testing

    Args:
        images_dir (str or pathlib.Path): path of directory containing images
        img_size (tuple): Default (256, 800) image size seen by the network
        transform : torchvision transformations that will tensorize and resize the 
                    input image
    """

    def __init__(self, images_dir, img_size=(256, 800), transform=None, fileformat='.tiff'):
        super(BarcodeTestDataset, self).__init__()
        self.images_dir = Path(images_dir) if isinstance(images_dir, str) else images_dir
        assert isinstance(self.images_dir, pathlib.Path) is True, "data dir is not a valid path"

        self.img_size = img_size
        
        self.transform = transform
        self.fileformat = fileformat
        # build filenames
        self.filenames = list(self.images_dir.glob('*' + self.fileformat))
        
        self.batch_count = 0

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image_filename = self.filenames[idx]
        image = imread(image_filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        datapoint = {
            'image': image
        }
        if self.transform is not None:
            transformed = self.transform(datapoint)
            image = transformed['image']
        return {
            'image': image,
            'path': str(image_filename)
        }

    def collate_fn(self, batch):
        self.batch_count += 1
        
        # drop invalid images
        batch = [data for data in batch if data is not None]
        
        #image stack
        images = torch.stack([data['image'] for data in batch])
        
        # paths stack
        paths = [data['path'] for data in batch]
        
        return paths, images

    def plot_datapoint(self, idx):
        datapoint = self.__getitem__(idx)
        image = datapoint['image']
        
        if image.ndim == 3 and image.shape[2] == 3:
            image = image[:, :, 0]
        if image.ndim == 3 and (image.shape[0] == 3 or image.shape[0] == 1):
            image = image[0, :, :]
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image, cmap='gray')
        height, width = image.shape
        plt.show()