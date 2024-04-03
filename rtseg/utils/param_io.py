

# File conatains classes to parse parameters
# Most of the code is copied and modified from
# https://github.com/TuragaLab/DECODE/blob/master/decode/utils/param_io.py


import json
import pathlib
import yaml
from typing import Union

import importlib.resources  as pkg_resources

from .types import RecursiveNamespace

class ParamHandling:
    file_extensions = ('.json', '.yml', '.yaml')

    def __init__(self):

        self.params_dict = None
        self.params_dot = None

    def _check_return_extension(self, filename):
        """
        Checks the specified file suffix as to whether it is in the allowed list 

        Args:
            filename
        
        Returns:
            file extension (str)
        """
        extension = pathlib.PurePath(filename).suffix
        if extension not in self.file_extensions:
            raise ValueError(f"Filename must be in {self.file_extensions}")
        
        return extension
    
    def load_params(self, filename: str | pathlib.Path, ref_type: str = 'seg') -> RecursiveNamespace:
        """
        Load parameters from file

        Args:
            filename (str): filename of the parameter file
            ref_type (str): reference type command to fill in the defaults 
                    if they are explicity written down your parameter file
                    'cellseg' or 'barcodedetect'

        Returns:
            an instance of RecursiveNamespace with all parameters loaded
            from the file
        """
        extension = self._check_return_extension(filename)
        if extension == '.json':
            with open(filename) as json_file:
                params_dict = json.load(json_file)
        elif extension in ('.yml', '.yaml'):
            with open(filename) as yaml_file:
                params_dict = yaml.safe_load(yaml_file)


        params_ref  = load_reference(ref_type)        
        params_dict = autofill_dict(params_dict, params_ref)
        params = RecursiveNamespace(**params_dict)

        self.params_dict = params_dict
        self.params_dot = params

        return params
    
    def write_params(self, filename: Union[str, pathlib.Path],
                      param: Union[dict, RecursiveNamespace]):
        """
        Write parameter file to path

        Args:
            filename (str | pathlib.Path): filename to write
            param (dict | RecursiveNamespace): parameters to write

        """
        filename = filename if isinstance(filename, pathlib.Path) else pathlib.Path(filename)

        extension = self._check_return_extension(filename)

        if isinstance(param, RecursiveNamespace):
            param = param.to_dict()

        # Create directory if it doesn't exist
        p = pathlib.Path(filename)
        try:
            # create all directories in the path specified, if they don't exist
            pathlib.Path(p.parents[0]).mkdir(parents=True, exist_ok=True)
        except FileNotFoundError:
            raise FileNotFoundError("Couldn't create directory to write params")

        if extension == '.json':
            with filename.open('w') as write_file:
                json.dump(param, write_file, indent=4)
        elif extension in ('.yml', '.yaml'):
            with filename.open('w') as yaml_file:
                yaml.dump(param, yaml_file)

    def convert_param_file(self, file_in: Union[str, pathlib.Path], 
                           file_out: Union[str, pathlib.Path]):
        """
        Simple wrapper to convert file from and to json / yaml
        Args:
            file_in: input file
            file_out: output file
        """
        params = self.load_params(file_in)
        self.write_params(file_out, params)

def load_reference(ref_type: str = 'cellseg'):
    """
    Loads the reference params file, that should contain all the parameters
    ever used

    Args:
        ref_type (str): 'cellseg' or 'barcodedetect'

    Returns
        params_ref (dict)

    """
    if ref_type == 'cellseg':
        from ..cellseg import configs
        param_ref = pkg_resources.open_text(configs, 'cellseg.yaml')
        param_ref = yaml.load(param_ref, Loader=yaml.SafeLoader)
    
    elif ref_type == 'barcodedetect':
        from ..barcodedetect import configs # type: ignore
        param_ref = pkg_resources.open_text(configs, 'barcodedetect.yaml')
        param_ref = yaml.load(param_ref, Loader=yaml.SafeLoader)

    return param_ref



def load_params(file, ref_type: str = 'cellseg'): # alias
    return ParamHandling().load_params(file, ref_type=ref_type)

def save_params(file, param):
    ParamHandling().write_params(file, param)


def autofill_dict(x: dict, reference: dict, mode_missing: str = 'include') -> dict:
    """
    Fill dict 'x' with keys and values of reference if they are not present in x 

    Args:
        x: input dict to be filled
        reference: reference dictionary
        mode_missing: 'inclue or 'exclude' mode for missing values
    """
    if mode_missing == 'exclude': # create a new dict and copy
        out = {}
    elif mode_missing == 'include':
        out = x
    else:
        raise ValueError

    for k, v in reference.items():
        if isinstance(v, dict):
            out[k] = autofill_dict(x[k] if k in x else {}, v)
        elif k in x: 
            out[k] = x[k]
        else:
            out[k] = reference[k]
    
    return out