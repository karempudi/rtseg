
import torch
import numpy as np
import edt # type: ignore
import pathlib
from pathlib import Path
from skimage.io import imread
from tqdm import tqdm

from rtseg.cellseg.numerics.differentiation.finite_differences import gaussian_smoothed_finite_differences


# This file generates the vector fields required by computing
# the derivatives of the signed distance transforms or smoothed
# signed distance transforms


def mask_bounding_box(mask):
    """
    Returns bounding box given a binary mask. Bounding box given in the form:

        [x_0, y_0, x_1, y_1]

    Where (x_0, y_0) is the top left pixel of the box and (x_1, y_1) is the
    bottom right pixel of the box. To index a numpy array do:

        np.array[y_0 : y_1, x_0 : x_1]

    Args:
        mask (np.ndarray): Binary Mask.

    Returns:
        list: Bounding box of binary mask.

    """
    non_zero_idx = np.where(mask == 1)

    x = non_zero_idx[1]
    y = non_zero_idx[0]

    bounding_box = [
        np.min(x),
        np.min(y),
        np.max(x) + 1,
        np.max(y) + 1
    ]

    return bounding_box

def compute_sdf(semantic_mask):
    """
    Given a semantic mask of shape (H, W) this function will compute the signed
    distance of each piel to the border of the mask. Used for the computation
    of the vector fields. 

    Args:
        semantic_mask (np.ndarray): The boolean semantic mask.

    Returns:
        np.ndarray: The SDF of the semantic_mask. Of shape: (H, W).

    """
    sdf = edt.sdf(
        semantic_mask, 
        order = 'C',
        parallel = -1 
    )

    return sdf

def sdf_vector_field(labels, kernel_size, alpha=10, device="cpu"):
    """
    Computes the vector field as proposed in the "On Vector Fields for Instance
    Segmentation" article. This is done by computing the signed distance field
    (SDF) for each mask in `labels`, then computing the gradient via a large
    gaussian-smoothed finite differences kernel.

    Unfortunately there isn't a PyTorch implementation of the SDF (that I know
    of) so we must convert tensors to NumPy arrays first, then run the SDF
    computation on the CPU.

    Args:
        labels (torch.Tensor): The instance segmentation labels in the form of
            0 (background), 1 (first instance mask), 2 (second instance mask), ...,
            n (n'th instance mask). Shape must be (1, H, W).

        kernel_size (int): The size of the finite difference gaussian kernel
            used to compute the numeric gradient of the SDF. 

        alpha (int): The truncation value used on the SDF. 

    Returns:
        torch.Tensor: The vector field of shape (2, H, W).

    """
    _, H, W = labels.shape
    pad     = kernel_size // 2

    unique = torch.unique(labels)[1:]
    np_labels = labels[0].detach().cpu().numpy().astype(np.uint32)

    vector_field = torch.zeros((2, H, W), device = device)
    for i in unique:
        curr = np_labels == int(i)

        x_0, y_0, x_1, y_1 = mask_bounding_box(curr)

        trimmed_curr = curr[y_0 : y_1 + 1, x_0 : x_1 + 1]

        padded_curr = np.pad(trimmed_curr, ((pad, pad), (pad, pad)))

        sdf = compute_sdf(padded_curr)
        sdf[sdf > alpha] = alpha

        # Use this as an example of why smoothing is important. 
#        out = np.stack(np.gradient(sdf), axis = 0)[None]

        # Compute the gradient. 
        out = gaussian_smoothed_finite_differences(
            torch.Tensor(sdf[None][None]).to(device), 
            kernel_size = kernel_size, 
            device = device
        )[0]

        vector_field[:, curr] = out[:, padded_curr]

    return vector_field.to(device)


def save_sdf_vf(labels_dir, save_dir, kernel_size: int, alpha: int = 10,
                labels_ext: str = '.png', vf_delimiter: str = "_vf"):
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

        alpha (int): The truncation value used on the SDF. 

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

    for i, filename in tqdm(enumerate(labels_imgs_filenames, 0)):
        # read image 
        image = imread(filename).astype('float32')
        labels = torch.from_numpy(image)[None, :]
        # compute sdf vf
        vf = sdf_vector_field(labels, kernel_size=kernel_size, alpha=alpha, device="cpu") 
        # save the vf to file
        save_filename = save_dir / Path(filename.stem + vf_delimiter + '_' + str(kernel_size) + ".npy")
        with open(save_filename, "wb") as f:
            np.save(f, vf.numpy())
    return None

if __name__ == "__main__":
    # plot vector fields of an image in the resources direcotry
    import matplotlib.pyplot as plt # type: ignore
    fig, ax = plt.subplots()
    plt.show()
