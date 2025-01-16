
from typing import Tuple

def get_device_str(device: str) -> Tuple[str, int]:
    """
    Takes a string and splits it into device and 
    device index.
    For ex: cuda:1 --> cuda, 1
    """
    if device != 'cpu' and device[:4] != 'cuda':
        raise ValueError("Hardware device is not set correctly")

    if device == 'cpu':
        return 'cpu', None
    elif len(device) == 4:
        return 'cuda', None
    else:
        return 'cuda', int(device.split(':')[-1])