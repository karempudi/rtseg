
import pathlib
import numpy as np
from skimage.io import imread
from torch.utils.data import Dataset
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt # type: ignore


class PhaseContrast(Dataset):
    """
    Directory structure of the phase contrast dataset

    """

    def __init__(self, phase_dir, labels_dir , vf_dir = None, vf: bool = False, 
                labels_delimiter: str = '_masks', vf_delimiter : str = '_vf', transforms = None,
                phase_format: str ='.png', labels_format: str ='.png', vf_format: str = '.npy'):
        """
        Dataloader for gettting training/ testing datasets

        Args:
            phase_dir (str | pathlib.Path): directory of phase images

            labels_dir (str | pathlib.Path): directory of labels

            vf_dir (str | pathlib.Path):  directory of vector fields
            
            vf (bool): If true, data loader will return phase image, labelled image and vector field tensor  

            labels_delimiter (str):

            vf_delimiter (str):

            transforms: 

            phase_format (str): 

            labels_format (str):
            
            vf_format (str):

        """
            
        if isinstance(phase_dir, str):
            self.phase_dir = Path(phase_dir)
        self.phase_dir = phase_dir
        assert isinstance(self.phase_dir, pathlib.Path), "`phase_dir` needs to be (str | pathlib.Path)"

        if isinstance(labels_dir, str):
            self.labels_dir = Path(labels_dir)
        self.labels_dir = labels_dir
        assert isinstance(self.labels_dir, pathlib.Path), "`labels_dir` needs to be (str | pathlib.Path)"

        if isinstance(vf_dir, str):
            self.vf_dir = Path(vf_dir)
        self.vf_dir = vf_dir
        assert isinstance(self.vf_dir, pathlib.Path), "`vf_dir` needs to be (str | pathlib.Path)"

        self.transforms = transforms
        
        self.phase_filenames = list(self.phase_dir.glob("*" + phase_format))

        # construction using phase_filenames
        self.labels_filenames: List[pathlib.Path] = [ labels_dir / Path(filename.stem + labels_delimiter + labels_format)
                                                             for filename in self.phase_filenames]

        self.vf = False
        if vf:
            self._getitem = self._get_image_mask_vf
            self.vf = vf
            # construction using phase_filenames
            self.vf_filenames: List[pathlib.Path] = [ vf_dir / Path(filename.stem + vf_delimiter + vf_format)
                                                            for filename in self.labels_filenames]

        else:
            self._getitem = self._get_image_mask

    def __len__(self):
        return len(self.phase_filenames)

    def _get_image_mask_vf(self, idx):
        phase_filename = self.phase_filenames[idx]
        mask_filename = self.labels_filenames[idx]
        vf_filename = self.vf_filenames[idx]

        image = imread(phase_filename).astype(np.float32)
        mask = imread(mask_filename).astype(np.float32)
        vf = np.load(vf_filename).astype(np.float32)

        if self.transforms is not None:
            image, mask, vf = self.transforms(image, mask, vf)

        return image, mask, vf

    def _get_image_mask(self, idx):
        phase_filename = self.phase_filenames[idx]
        mask_filename = self.labels_filenames[idx]

        image = imread(phase_filename).astype(np.float32)
        mask = imread(mask_filename).astype(np.float32)

        if self.transforms is not None:
            image, mask = self.transforms(image, mask)
        
        return image, mask
   
    def __getitem__(self, idx):
        return self._getitem(idx)

    def plot_item(self, idx):
        if self.vf and (self.transforms is None):
            nrows, ncols = 2, 2
            fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
            image, mask, vf = self.__getitem__(idx)
            ax[0, 0].imshow(image, cmap='gray')
            ax[0, 0].set_title('Phase contrast')
            ax[0, 1].imshow(mask)
            ax[0, 1].set_title('Mask')
            ax[1, 0].imshow(vf[0])
            ax[1, 0].set_title('vf_x')
            ax[1, 1].imshow(vf[1])
            ax[1, 1].set_title('vf_y')
            fig.suptitle(f'{self.phase_filenames[idx].name}')
            plt.show()
        elif (self.transforms is None):
            nrows, ncols = 1, 2
            fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
            image, mask = self.__getitem__(idx)
            ax[0].imshow(image, cmap='gray')
            ax[0].set_title('Phase contrast')
            ax[1].imshow(mask)
            ax[1].set_title('Mask')
            fig.suptitle(f'{self.phase_filenames[idx].name}')
            plt.show()


