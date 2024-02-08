
import torch

def init_values_mesh(H, W, device = "cpu"):
    y, x = torch.meshgrid(
        torch.arange(0, H, device = device), 
        torch.arange(0, W, device = device),
        indexing = "ij"
    )

    init_values = torch.stack([x, y], dim = 0)

    return init_values


def init_values_mesh_batched(B, H, W, device = "cpu"):
    init_values = init_values_mesh(H, W, device)

    init_values = init_values.repeat(B, 1, 1, 1).float()

    return init_values


def init_values_semantic(semantic, device):
    B, _, H, W = semantic.shape

    init_values = init_values_mesh(H, W, device)

    init_values = init_values[:, semantic[0][0]].float()

    return init_values
