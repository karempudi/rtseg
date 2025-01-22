
# The code for this file is mostly taken from 
# https://github.com/TuragaLab/DECODE/blob/master/decode/neuralfitter/utils/logger.py
# and modified to fit our training purposes
import time
import matplotlib.pyplot as plt
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

class NoLog(tensorboard.SummaryWriter):
    """
    The hardcoded No-Op of the tensorboard SummaryWriter.
    """

    def __init__(self, *args, **kwargs):
        return

    def add_scalar(self, *args, **kwargs):
        return

    def add_scalars(self, *args, **kwargs):
        return

    def add_scalar_dict(self, *args, **kwargs):
        return

    def add_histogram(self, *args, **kwargs):
        return

    def add_figure(self, tag, figure, *args, **kwargs):
        plt.close(figure)
        return

    def add_figures(self, *args, **kwargs):
        return

    def add_image(self, *args, **kwargs):
        return

    def add_images(self, *args, **kwargs):
        return

    def add_video(self, *args, **kwargs):
        return

    def add_audio(self, *args, **kwargs):
        return

    def add_text(self, *args, **kwargs):
        return

    def add_graph(self, *args, **kwargs):
        return

    def add_embedding(self, *args, **kwargs):
        return

    def add_pr_curve(self, *args, **kwargs):
        return

    def add_custom_scalars(self, *args, **kwargs):
        return

    def add_mesh(self, *args, **kwargs):
        return

    def add_hparams(self, *args, **kwargs):
        return

class DictLogger(NoLog):
    """
    Simple logger that can log scalars to a dictionary
    """

    def __init__(self):
        super().__init__()
        self.log_dict = {}

    def add_scalar_dict(self, prefix: str, scalar_dict: dict, global_step=None, walltime=None):
        for name, value in scalar_dict.items():
            self.add_scalar(prefix + name, value, global_step=global_step, walltime=walltime)
    
    def add_scalar(self, prefix: str, scalar_value: float, global_step=None, walltime=None):

        if walltime is None:
            walltime = time.time()

        if prefix in self.log_dict:
            if global_step is None:
                # I don't understand this line of code yet, 
                # where did the key 'global_step' get intialized at in decode
                # So I initialized it down below
                global_step = self.log_dict['global_step'] + 1
            
            self.log_dict[prefix]['scalar'].append(scalar_value)
            self.log_dict[prefix]['step'].append(global_step)
            self.log_dict[prefix]['walltime'].append(walltime)

        else:
            # do the initialzation of the values
            if global_step is None:
                global_step = 0 
                self.log_dict['global_step'] = global_step
            
            val_init = {
                'scalar': [scalar_value],
                'step': [global_step],
                'walltime': [walltime]
            }
            self.log_dict.update({
                prefix: val_init
            })


class MultiLogger:
    """
    A 'Meta-Logger', i.e. a logger that calls its componenets.
    Note all component loggers are assumed to have the same methdos.
    """

    def __init__(self, logger):

        def execute_for_all(loggers, method: str):
            """
            Execute all loggers and put the results in a list
            after executing them sequentially.
            """
            def idk(*args, **kwargs):
                # for l in loggers
                return [getattr(logger, method)(*args, **kwargs) for logger in loggers]

            return idk


        self.logger = logger

        # get all methods from the 0th logger
        methods = [method_name for method_name in dir(self.logger[0]) if callable(getattr(self.logger[0], method_name))]
        methods = [m for m in methods if '__' not in m]

        for m in methods:
            setattr(self, m, execute_for_all(self.logger, m))

