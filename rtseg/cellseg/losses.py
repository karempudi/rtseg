
import torch
import torch.nn as nn
import torch.nn.functional as F
from rtseg.cellseg.numerics.interpolation.interpolate_vf import interpolate_vf
from rtseg.cellseg.numerics.integration.integrate_vf import ivp_solver


# Dice loss is mostly used for channel segmentation in our nets
class DiceLoss(nn.Module):
    """
    Binary dice loss for semantic segmentation. This code is reworked from
    these GitHub repos:
        - https://github.com/qubvel/segmentation_models.pytorch
        - https://github.com/BloodAxe/pytorch-toolbelt

    """
    def __init__(self, from_logits = True, log_loss = False, smooth = 0.0, eps = 1e-7):
        """
        Args:
            log_loss (bool): If True, the loss is computed as `-log(dice_coeff)`,
                otherwise `1 - dice_coeff`.

            from_logits (bool): If True, assumes y_pred are raw logits.
            smooth (float): Smoothness constant for dice coefficient.
            eps (float): For numerical stability to avoid zero division error.

        """
        super(DiceLoss, self).__init__()

        self.from_logits = from_logits
        self.log_loss = log_loss
        self.smooth = smooth
        self.eps = eps

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred (torch.Tensor): Of shape (B, C, H, W).
            y_true (torch.Tensor): Of shape (B, C, H, W).

        Returns:
            torch.Tensor: The loss.

        """
        assert y_true.size(0) == y_pred.size(0), "`y_true` and `y_pred` have different shapes, check shapes :("

        if self.from_logits:
            y_pred = F.logsigmoid(y_pred).exp()

        bs   = y_true.size(0)
        dims = (0, 2)

        y_true = y_true.view(bs, 1, -1)
        y_pred = y_pred.view(bs, 1, -1)

        scores = self._compute_score(
            y_pred, 
            y_true.type_as(y_pred), 
            smooth = self.smooth, 
            eps = self.eps, 
            dims = dims
        )

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        mask = y_true.sum(dims) > 0
        loss = loss * mask.to(loss.dtype)

        return self._reduction(loss)

    def _reduction(self, loss):
        return loss.mean()

    def _compute_score(self, y_pred, y_true, smooth = 0.0, eps = 1e-7, dims = ()):
        assert y_pred.size() == y_true.size()

        intersection = torch.sum(y_pred * y_true, dim = dims)
        cardinality = torch.sum(y_pred + y_true, dim = dims)
        print(intersection, cardinality)

        dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)

        return dice_score

class TverskyLoss(DiceLoss):
    """
    This code is reworked from this GitHub repo:
        - https://github.com/qubvel/segmentation_models.pytorch
        - https://github.com/BloodAxe/pytorch-toolbelt

    Tversky loss for semantic segmentation. Notice this class inherits
    `DiceLoss` and adds a weight to the value of each TP and FP given by
    constants alpha and beta. With alpha == beta == 0.5, this loss becomes
    equal to the Dice loss. `y_pred` and `y_true` must be torch tensors of
    shape (B, C, H, W).

    """
    def __init__(self, from_logits = True, log_loss = False, smooth = 0.0, eps = 1e-7, 
                 alpha = 0.5, beta = 0.5, gamma = 1.0):
        """
        Args:
            from_logits (bool): If True, assumes y_pred are raw logits.
            log_loss (bool): If True, the loss is computed as `-log(dice_coeff)`,
                otherwise `1 - dice_coeff`.

            smooth (float): Smoothness constant for dice coefficient.
            eps (float): For numerical stability to avoid zero division error.
            alpha (float): Weight constant that penalize model for FPs.
            beta (float): Weight constant that penalize model for FNs.
            gamma (float): Constant that squares the error function. Defaults to `1.0`.

        """
        super(TverskyLoss, self).__init__(from_logits, log_loss, smooth, eps)

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _reduction(self, loss):
        return loss.mean() ** self.gamma

    def _compute_score(self, y_pred, y_true, smooth = 0.0, eps = 1e-7, dims = ()):
        assert y_pred.size() == y_true.size()

        intersection = torch.sum(y_pred * y_true, dim = dims)  
        fp = torch.sum(y_pred * (1.0 - y_true), dim = dims)
        fn = torch.sum((1 - y_pred) * y_true, dim = dims)

        tversky_score = (intersection + smooth) / (intersection + self.alpha * fp + self.beta * fn + smooth).clamp_min(eps)

        return tversky_score
 
# This loss is used on the vector fields
class IVPLoss(nn.Module):
    """
    The initial value problem loss function as described in the "On Vector
    Fields for Instance Segmentation" article. This loss function aims to
    minimize the distance between trajectories of points under integration
    through the ground truth and predicted vector field. 

    What this means is 

    Known issues:
        - Under interpolation we clip values that go beyond our image domain,
        therefore backpropogated gradients at the image borders might not be as
        expected. This isn't necessarily a problem with this loss function
        iteself as it is with the interpolation.

    """
    
    def __init__(self, dx: float = 0.5, n_steps: int = 8, 
                    solver: str = 'euler', mode: str = 'bilinear_batched',
                    device: str | torch.device = 'cpu'):
        """
        Args:
            dx (float): Numeric integration step size.
            n_steps (int): Number of integration steps.
            solver (str): Numeric integration solver. One of:
                - "euler"
                - "midpoint"
                - "runge_kutta"
            mode (str): The type of interpolation to do. One of:
                - "bilinear_batched"
                - "nearest_batched"
            device (str): hardware device

        """
        super(IVPLoss, self).__init__()
        self.dx = dx 
        self.n_steps = n_steps
        self.solver = solver
        self.mode = mode
        self.device = device

    def _compute_init_values(self, shape):
        B, C, *dims = shape

        coords = [torch.arange(0, dim_value, device = self.device) for dim_value in dims]
        mesh = torch.meshgrid(coords, indexing = "ij")

        init_shape = [B, 1] + ([1] * len(dims))
        init_values = torch.stack(mesh[::-1], dim = 0)
        init_values = init_values.repeat(init_shape)

        return init_values 

    def _compute_batched_trajectories(self, vf):
        """
        Takes a vector field and computes the trajectories of
        the points in the vector field using the ivp_solver.
        The solver stores all the intermediate solutions returns
        a stack of tensors that show how different initial points 
        move when the field is applied on them. This function
        essentially does the batched version of integration steps, but
        will keep the paths taken so that both ground truth and
        predicted vector fields take the same paths essentially when
        training is done.

        Args:
            vf (torch.Tensor): of shape (B, 2, H, W)
        
        """
        init_values = self._compute_init_values(vf.shape)

        # returns an interpolated vf, that you can call with points
        vf = interpolate_vf(vf, mode=self.mode)

        # store solutions as you do MSE between the trajectories taken
        trajectories = ivp_solver(vf,
                                  init_values=init_values,
                                  dx=self.dx,
                                  n_steps=self.n_steps,
                                  solver=self.solver,
                                  store_solutions=True)

        return trajectories

    def forward(self, vf_pred: torch.Tensor, vf_true: torch.Tensor):
        """
        Args:
            vf_pred (torch.Tensor): Of shape (B, 2, H, W)
            vf_true (torch.Tensor): Of shape (B, 2, H, W)

        Returns:
            torch.Tensor: The loss
        """
        true_trajectories = self._compute_batched_trajectories(vf_true)
        pred_trajectories = self._compute_batched_trajectories(vf_pred)

        loss_batch = F.mse_loss(true_trajectories, pred_trajectories)

        return loss_batch

