import torch

def test_torch():
    assert torch.__version__.startswith('2.1.2')