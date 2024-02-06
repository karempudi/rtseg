from rtseg.cellseg.network import SegNet
import torch

def test_net_input_outputs():
    net = SegNet(channels_by_scale=[1, 32, 64, 128],
                 num_outputs=[1, 2, 1])
    image = torch.randn((1, 1, 1024, 1024))
    semantic_cells, vf_cells, semantic_channels = net(image)
    assert semantic_cells.shape == (1, 1, 1024, 1024)
    assert semantic_channels.shape == (1, 1, 1024, 1024)
    assert vf_cells.shape == (1, 2, 1024, 1024)