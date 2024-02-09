
import pathlib
import numpy as np
from skimage.io import imread
from torch.utils.data import Dataset
from pathlib import Path
from typing import List


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
        assert isinstance(self.phase_dir, pathlib.Path), "`phase_dir` needs to be (str | pathlib.Path)"

        if isinstance(labels_dir, str):
            self.labels_dir = Path(labels_dir)
        assert isinstance(self.labels_dir, pathlib.Path), "`labels_dir` needs to be (str | pathlib.Path)"

        if isinstance(vf_dir, str):
            self.vf_dir = Path(vf_dir)
        assert isinstance(self.vf_dir, pathlib.Path), "`vf_dir` needs to be (str | pathlib.Path)"

        self.transforms = transforms
        
        self.phase_filenames = list(self.phase_dir.glob("*" + phase_format))

        # construction using phase_filenames
        self.labels_filenames: List[str | pathlib.Path] = []

        if vf:
            self._getitem = self._get_image_mask_vf
            # construction using phase_filenames
            self.vf_filenames: List[str | pathlib.Path] = []

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

    def _get_image_mask(self, idx):
        phase_filename = self.phase_filenames[idx]
        mask_filename = self.labels_filenames[idx]

        image = imread(phase_filename).astype(np.float32)
        mask = imread(mask_filename).astype(np.float32)

        if self.transforms is not None:
            image, mask = self.transforms(image, mask)
   
    def __getitem__(self, idx):
        return self._getitem(idx)

