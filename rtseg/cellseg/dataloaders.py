
from torch.utils.data import Dataset
from pathlib import Path


class PhaseContrast(Dataset):
    """
    Dataloader for gettting training/ testing datasets

    Args:
        data_dir (str | pathlib.Path): path of the direcory containing data

        split (str): Options are 
            "train" --> 
            "test" --> 
        
        vf (bool): If true 

        vf_delimiter (str):

        transforms: 

    """
    def __init__(self, data_dir, split = "train", vf: bool = False, 
                 vf_delimiter: str = '_vf', transforms = None):
        
        self.data_dir = Path(data_dir)
        self.transforms = transforms

    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass

