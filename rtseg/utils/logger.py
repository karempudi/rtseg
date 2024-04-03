
from torch.utils import tensorboard


class SummaryWriter(tensorboard.SummaryWriter):

    def __init__(self, filter_keys=(), *args, **kwargs):
        """
        Args:
            filer_keys: keys to be filtered in add_scalar_dict method
            *args:
            **kwargs:
        """
        super().__init__(*args, **kwargs)

        self.filter_keys = filter_keys

    def add_scalar_dict(self, prefix: str, scalar_dict: dict, global_step=None, walltime=None):
        """
        Adds a couple of scalars that are in dictionary to the summary.
        Note that this is different from 'add_scalars'
        """

        for name, value in scalar_dict.items():
            # basically ignore stuff in the filter
            if name in self.filter_keys:
                continue

            self.add_scalar(prefix + name, value, global_step=global_step, walltime=walltime)
