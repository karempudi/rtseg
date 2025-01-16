
import torch.utils.tensorboard

class SummaryWriter(torch.utils.tensorboard.SummaryWriter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_scalar_dict(self, prefix: str, scalar_dict: dict, global_step=None, walltime=None):
        """
        Adds all the scalars that are passed as a dictionary to the logger 
        under the prefix
        """
        for name, value in scalar_dict.items():
            self.add_scalar(prefix + name, value, global_step=global_step, walltime=walltime)

