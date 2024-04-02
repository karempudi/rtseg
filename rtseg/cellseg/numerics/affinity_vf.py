
import pathlib
from pathlib import Path
import torch.nn.functional as F
import torch
import numpy as np
from skimage.io import imread


def affinity_vector_field(instance_mask, kernel_size, device = "cpu"):
    """
    Computes the ground truth affinity vector field given an instance
    mask.

    Kernel sizes from 11 - 19 seem to work well on the BPCIS (phase contrast) dataset. 

    Args:
        instance_mask (torch.Tensor): Of shape (B, C, H, W).

    Returns:
        (torch.Tensor): The vector field that is to be predicted by your 
            model. Will be of shape (B, 2, H, W)

    """
    B, _, H, W = instance_mask.shape

    semantic = instance_mask.clone() > 0
    sem_idx  = semantic.repeat(1, 2, 1, 1)

    target_kernel = apply_ones_kernel(instance_mask, kernel_size, gradient = False)
    target_kernel = target_kernel.reshape(B, kernel_size, kernel_size, H, W)

    vector_field = affinity_to_vf(target_kernel, device = device)
    vector_field[~sem_idx] = 0.0

    return vector_field


def apply_ones_kernel(x, kernel_size, gradient = False):
    B, C, H, W = x.shape
    mid = kernel_size // 2
    
    out = neighbor_view(x, kernel_size, padding_int = -1)

    if not gradient:
        out = out.reshape(B, kernel_size, kernel_size, H, W)

        instance_values = out[:, mid, mid].reshape(B, 1, 1, H, W)

        out = torch.where(out != instance_values, 0, 1)

        out = out.reshape(B, kernel_size * kernel_size, H, W)

    return out


def neighbor_view(x, n, padding_int = -1):
    """
    Uses PyTorch's im2col to get a n^2 convolution style neighbor 
    view of a 2D array. 

    Args:
        x (torch.Tensor): 4D tensor input of shape (B, C, H, W). 
        n (int): The number of n^2 neighbors. Must be an odd 
            number.

    Returns:
        torch.Tensor: The n^2-neighbor view of the image of
            shape (1, n, n, H, W).

    """
    x_dtype = x.dtype
    mid     = n // 2

    x = F.pad(x.float(), (mid, mid, mid, mid), "constant", padding_int)

    out = F.unfold(x.float(), kernel_size = (n, n)).to(x_dtype)

    return out


def affinity_to_vf(target, device = "cpu"):
    """
    Given an affinity prediction of shape (B, K_H, K_W, H, W). This 
    will return the vector field.

    Args:
        target (torch.Tensor): This is the affinity neighbor view of
            your ground truth instance segmentation. This tensor must
            be of shape (B, K_H, K_W, H, W).

    Returns:
        vf (torch.Tensor): Of shape (2, H, W).
    
    """
    B, K_H, K_W, H, W = target.shape
    
    x_target = target.sum(axis = 1).reshape(B, K_W, H, W).permute(0, 2, 3, 1) / K_H
    y_target = target.sum(axis = 2).reshape(B, K_H, H, W).permute(0, 2, 3, 1) / K_H

    vec_product_l =  torch.ones(K_H // 2)
    vec_product_r = -torch.ones(K_H // 2)
    vec_product = torch.cat([vec_product_l, torch.Tensor([0]), vec_product_r], axis = 0)
    vec_product = vec_product.reshape(K_H, 1).to(device)

    vf_x = -torch.matmul(x_target.float(), vec_product.float()).permute(0, 3, 1, 2) / 2 
    vf_y = -torch.matmul(y_target.float(), vec_product.float()).permute(0, 3, 1, 2) / 2

    vf = torch.cat([vf_x, vf_y], axis = 1)

    return vf.float()


def save_affinity_vf(labels_dir, save_dir, kernel_size: int,
                labels_ext: str = '.png', vf_delimiter: str = "_af"):
    """
    For each image in the labels_dir, calculate vector fields and save them as
    .npy files int the save_dir.

    Args:
        labels_dir (str | pathlib.Path): directory containing labels images where
            each image has cells labelled (1, 2, 3, ... N) and background 0.

        save_dir (str | pathlib.Path): director to save *.npy vector field file
            for each of the image in the labels_dir. If an image has shape (H, W),
            it's vector field will have shape (2, H, W)

        kernel_size (int): The size of the finite difference gaussian kernel
           used to compute the numeric gradient of the SDF. 

        labels_ext (str): file extension of the labels files. '.png' | '.tiff' | '.tif' 

        vf_delimiter (str): append the vector field files with a delimiter when
                        saving. actual filename will be saved as
                        'labels_vf_delimiter_kernel_size.npy'
    Returns:
        None
    """
    if isinstance(labels_dir, str):
        labels_dir = Path(labels_dir)
    assert isinstance(labels_dir, pathlib.Path), "`labels_dir` should be str | pathlib.Path"

    if isinstance(save_dir, str):
        labels_dir = Path(save_dir)
    assert isinstance(save_dir, pathlib.Path), "`save_dir` should be str | pathlib.Path"

    labels_imgs_filenames = list(labels_dir.glob("*" + labels_ext))
    labels_imgs_filenames = sorted(labels_imgs_filenames)

    for i, filename in enumerate(labels_imgs_filenames, 0):
        # read image 
        image = imread(filename).astype('float32')
        labels = torch.from_numpy(image)[None, None, :]
        # compute sdf vf
        vf = affinity_vector_field(labels, kernel_size=kernel_size, device="cpu") 
        # save the vf to file
        save_filename = save_dir / Path(filename.stem + vf_delimiter + '_' + str(kernel_size) + ".npy")
        with open(save_filename, "wb") as f:
            np.save(f, vf.numpy()[0])
        print(f"Completed {i}/ {len(labels_imgs_filenames)} -- {filename.name}", end='\r', flush=True)
    return None