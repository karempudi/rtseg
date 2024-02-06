

import torch
import torch.nn.functional as F

from rtseg.cellseg.numerics.differentiation.kernels import finite_difference_kernel, finite_gaussian_kernel

def gaussian_smoothed_finite_differences(x, kernel_size, device = "cpu"):
    gaussian_x, gaussian_y = finite_gaussian_kernel(kernel_size)

    gaussian = torch.stack([gaussian_x, gaussian_y], dim = 0).to(device)

    out = F.conv2d(x, gaussian[:, None], padding = kernel_size // 2)

    return out


def finite_differences(x, kernel_size, device = "cpu"):
    finite_x, finite_y = finite_difference_kernel(kernel_size)

    finite = torch.stack([finite_x, finite_y], dim = 0).to(device)

    out = F.conv2d(x, finite[:, None], padding = kernel_size // 2)

    return out


