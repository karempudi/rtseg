
import numpy as np
import torch
from rtseg.cellseg.numerics.cluster import cluster
from rtseg.cellseg.numerics.interpolation.interpolate_vf import interpolate_vf
from rtseg.cellseg.numerics.integration.utils import init_values_semantic
from rtseg.cellseg.numerics.integration.integrate_vf import ivp_solver

def construct_mask(vf, semantic, dx: float = 0.1, n_steps: int = 5,
                   interpolation_mode: str ='bilinear', solver_integration: str = 'euler',
                   eps: float = 2.25, min_samples: int = 15, fast: bool = True, snap_noise: bool = True,
                   device: str ='cpu', store_solutions: bool = False, return_np: bool = False):
    """
    Args:
        vf (torch.Tensor): shape (1, 2, H, W) vector fields predicted by the network
        semantic (torch.Tensor): shape (1, 1, H, W) boolean tensor with True values
                    used as initial points to integrate the field over
        dx (float): step change to move the points in the direction of the field
        n_steps (int): number of steps, depends on the size of the cells and nature
                    of the field used.
        interpolation_mode (str):
        solver_integration (str): 'euler' or 'midpoint' or 'runge_kutta'
        eps (float): EPS for the DBSCAN clustering. The maximum radius to consider that two points
            are in a neighbourhood
        min_samples (int): number of points in the neighbourhood of a point to consider as a core point
            (refer: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
        fast (bool): if True uses fast dbscan to cluster or uses sklearn clustering
        snap_noise (bool): if true removes noisy labels and connects them to nearest neighbour
        device (str): cpu or cuda:0
        return_np (bool): return numpy array if True.

    Returns:
        labelled_image (torch.Tensor | np.ndarray): (N, 1, H, W) shaped labelled image

    """

    # check devices of the tensors
    if device == 'cpu':
        vf = vf.cpu()
        semantic = semantic.cpu()

    #B, _, H, W = vf.shape
    continuous_vf = interpolate_vf(vf, mode=interpolation_mode)

    #semantic = semantic.clone() > 0
    semantic = semantic > 0

    # initialize points to integrate over. These are the '1's in the
    # semantic segmentation mask
    init_values = init_values_semantic(semantic, device=device)

    solutions = ivp_solver(continuous_vf, init_values, dx, n_steps, solver=solver_integration,
                            store_solutions=store_solutions)[-1]
    #print(f"Solutions device: {solutions.device}")
    #print(f"Semantic device: ", {semantic.device})

    labelled_image = cluster(solutions, semantic, fast = fast, 
                        eps=eps, min_samples=min_samples, snap_noise=snap_noise) 
    

    if return_np:
        labelled_image = labelled_image.numpy()

    return labelled_image


def construct_masks_batch(vf, semantic, dx: float = 0.1, n_steps: int = 5,
                   interpolation_mode: str ='bilinear', solver_integration: str = 'euler',
                   eps: float = 2.25, min_samples: int = 15, fast: bool = True, snap_noise: bool = True,
                   device: str ='cpu', store_solutions: bool = False, return_np: bool = False):
    
    """
    Simply runs the construct_mask funciton on a batch of images, not parallelization

    TODO: Look for ways to parallelize this function if it is a performance bottleneck

    Args:
        vf (torch.Tensor): shape (N, 2, H, W) vector fields predicted by the network
        semantic (torch.Tensor): shape (N, 1, H, W) boolean tensor with True values
                    used as initial points to integrate the field over
        dx (float): step change to move the points in the direction of the field
        n_steps (int): number of steps, depends on the size of the cells and nature
                    of the field used.
        interpolation_mode (str):
        solver_integration (str): 'euler' or 'midpoint' or 'runge_kutta'
        eps (float): EPS for the DBSCAN clustering. The maximum radius to consider that two points
            are in a neighbourhood
        min_samples (int): number of points in the neighbourhood of a point to consider as a core point
            (refer: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
        fast (bool): if True uses fast dbscan to cluster or uses sklearn clustering
        snap_noise (bool): if true removes noisy labels and connects them to nearest neighbour
        device (str): cpu or cuda:0
        return_np (bool): return numpy array if True.

    Returns:
        labelled_image (torch.Tensor | np.ndarray): (N, 1, H, W) labelled image

    """
    B, _, _, _ = vf.shape

    labelled_masks = []
    for i in range(B):
        labelled_mask = construct_mask(vf[i][None, :], semantic[i][None, :],
                                       dx=dx, n_steps=n_steps,interpolation_mode=interpolation_mode,
                                       solver_integration=solver_integration, eps=eps,
                                       min_samples=min_samples,fast=fast, snap_noise=snap_noise,
                                       device=device, store_solutions=store_solutions,
                                       return_np=return_np)
        labelled_masks.append(labelled_mask)

    if return_np:
        return np.concatenate(labelled_masks)
    else:
        return torch.cat(labelled_masks, dim=0)
