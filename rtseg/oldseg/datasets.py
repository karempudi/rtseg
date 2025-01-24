import torch
import pathlib
from pathlib import Path
from torch.utils.data import Dataset
from typing import Union
from tifffile import imread


class MMDatasetUnetDual(Dataset):

    def __init__(self, data_dir: Union[str, pathlib.Path],
                transform=None, weights=False, fileformats = {
                    'phase': '.tif', 'cell_mask': '.tif', 'channel_mask': '.tif', 'weights': '.tif'
                }):
        """
        A dataset of a species is in data_dir, with subdirs 'phase', 'mask' & 'weights'
        If you want to include more species, use concatenation of different datasets 
        after creating an MMDatasetUnetDual for each species

        Dual means that we use one network to predict both cells and channels masks
        So that datasets and dataloaders should deliver identically transformed images
        of cell mask and channels mask for each transformed phase-contrast training image.
        If there are weigths, they should be specified for cells only, as we wont need weights
        for channels as they never really touch each other.

        Args:
            data_dir (str or pathlib.Path): directory containing data in directores,
                        'phase', 'cell_mask', 'channel_mask', 'weights_cell'(optional) 
                    For each phase file , there must be a cell mask file and channel mask file
                    and weights file (if used)
                    Note: Pass the extensions correctly for each dataset.
            transform: transforms applied to a datapoint in the dataset
            weights (boolean): are weights included int he dataset or not
            fileformat (dict): fileformats to grab files from directories with
        """
        super(MMDatasetUnetDual, self).__init__()

        self.data_dir =  data_dir if isinstance(data_dir, pathlib.Path) else Path(data_dir)
        self.phase_dir = self.data_dir / Path('phase')
        self.cell_mask_dir = self.data_dir / Path('mask')
        self.channel_mask_dir = self.data_dir / Path('channel_mask')
        self.use_weights = weights
        self.fileformats = fileformats
        self.transform = transform

        if self.use_weights:
            self.weights_dir = self.data_dir / Path('weights')

        self.phase_filenames = list(self.phase_dir.glob('*' + fileformats['phase']))
        self.cell_mask_filenames = [self.cell_mask_dir / Path(filename.stem + fileformats['cell_mask']) 
                                    for filename in self.phase_filenames]
        self.channel_mask_filenames = [self.channel_mask_dir / Path(filename.stem + fileformats['channel_mask']) 
                                    for filename in self.phase_filenames]
        if self.use_weights:
            self.weights_filenames = [self.weights_dir / Path(filename.stem + fileformats['weights']) 
                                    for filename in self.phase_filenames]


        self.batch_count = 0

    def __len__(self):
        return len(self.phase_filenames)

    def __getitem__(self, idx):

        phase_img = imread(self.phase_filenames[idx])
        cell_mask_img = imread(self.cell_mask_filenames[idx])
        channel_mask_img = imread(self.channel_mask_filenames[idx])
        if self.use_weights:
            weights_img = imread(self.weights_filenames[idx])
        else:
            weights_img = None
        
        height, width = phase_img.shape

        sample = {
            'phase': phase_img.astype('float32'),
            'mask': cell_mask_img,
            'channel_mask': channel_mask_img,
            'weights': weights_img,
            'filename': self.phase_filenames[idx].name,
            'raw_shape': (height, width)
        }

        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample
    
    def plot_datapoint(self, idx):
        pass

    def collate_fn(self, batch):
        self.batch_count += 1

        # drop invalid images
        batch = [data for data in batch if data is not None]

        phase = torch.stack([data['phase'] for data in batch])

        mask = []
        channel_mask = []
        weights = []
        for data in batch:
            if data['mask'] is not None:
                mask.append(data['mask'])
            else:
                mask.append(torch.tensor([-1]))
            
            if data['channel_mask'] is not None:
                channel_mask.append(data['channel_mask'])
            else:
                channel_mask.append('0')
            
            if data['weights'] is not None:
                weights.append(data['weights'])
            else:
                weights.append('0')

        if batch[0]['mask'] is not None:
            mask = torch.stack(mask)
        if batch[0]['channel_mask'] is not None:
            channel_mask = torch.stack(channel_mask)
        if batch[0]['weights'] is not None:
            weights = torch.stack(weights)

        filenames = [data['filename'] for data in batch]
        raw_shapes = [data['raw_shape'] for data in batch]

        return phase, mask, channel_mask, weights, filenames, raw_shapes


class MMDatasetUnetTest(Dataset):

    def __init__(self, images_dir: Union[str, pathlib.Path], fileformat='.tif*',
                transform=None):
        super(MMDatasetUnetTest, self).__init__()
        
        self.images_dir = images_dir if isinstance(images_dir, pathlib.Path) else Path(images_dir)
        self.fileformat = fileformat
        self.transform = transform

        self.filenames = list(self.images_dir.glob('*' + self.fileformat))

        self.batch_count = 0
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        
        phase_img = imread(self.filenames[idx]).astype('float32')
        height, width = phase_img.shape

        sample = {
            'phase': phase_img,
            'filename': self.filenames[idx].name,
            'raw_shape' : (height, width)
        }

        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample
    
    def plot_datapoint(self, idx):
        pass

    def collate_fn(self, batch):
        self.batch_count += 1

        # drop invalid images
        batch = [data for data in batch if data is not None]

        phase = torch.stack([data['phase'] for data in batch])

        filenames = [data['filename'] for data in batch]
        raw_shapes = [data['raw_shape'] for data in batch]

        return phase, filenames, raw_shapes
