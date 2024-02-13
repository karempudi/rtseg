
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
        labelled_image (torch.Tensor | np.ndarray): 

    """

    # check devices of the tensors

    #B, _, H, W = vf.shape
    continuous_vf = interpolate_vf(vf, mode=interpolation_mode)

    semantic = semantic.clone() > 0

    # initialize points to integrate over. These are the '1's in the
    # semantic segmentation mask
    init_values = init_values_semantic(semantic, device=device)

    solutions = ivp_solver(continuous_vf, init_values, dx, n_steps, solver=solver_integration,
                            store_solutions=store_solutions)[-1]

    labelled_image = cluster(solutions, semantic, fast = fast, 
                        eps=eps, min_samples=min_samples, snap_noise=snap_noise) 
    

    return labelled_image

def construct_masks_batch(vf):
    """
    Args:
        vf:
        semantic:
        dx:
        n_steps:
        interpolation_mode:
        solver_integration:
        eps:
        min_samples:
        fast:
        snap_noise:
        device:
        return_np:

    Returns:
        labelled_image (torch.Tensor | np.ndarray): 

    """
    return None
