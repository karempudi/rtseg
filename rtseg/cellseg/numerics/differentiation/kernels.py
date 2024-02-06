
import torch

# Kernels are filters that you apply. Convolve an image with 
# a kernel using torch conv2d to get appropriate filters applied

def mean_kernel(kernel_size):
    """
    Returns a kernel that is used to get the mean of the neighbourhood
    of a pixel
    Args:
        kernel_size (int): odd number.
    
    Returns:
        mean_kernel (torch.Tensor): a 3D tensor of shape (1, kernel_size, kernel_size)
    """
    assert kernel_size % 2, "`kernel_size` must be an odd number"

    ones_kernel = torch.ones((1, kernel_size, kernel_size))
    mean_kernel = ones_kernel / ones_kernel.numel()

    return mean_kernel      


def  gaussian_kernel(kernel_size, sigma=0.5):
    """
    Returns a kernel that is a can be used to apply a gaussian filter on a small
    neighbourhood of pixels (kernel_size // 2 away at most) to get a smoothed image

    When applied on the image it gives the same effect as skiamge.filters.gaussian

    Args:
        kernel_size (int): odd number, defines the neighbourhood over which smoothing
                is done
        sigma (float): default half a pixel, sigma defines how to weigh pixels in the
                neighbourhood
    
    Returns:
        gaussian_kernel (torch.Tensor): a 2d tensor of shape (kernel_size, kernel_size)

    """

    assert kernel_size % 2, "`kernel_size` must be an odd number"

    t    = torch.linspace(-1, 1, kernel_size)
    x, y = torch.meshgrid(t, t, indexing = "ij")
    dst  = torch.sqrt(x * x + y * y)

    gaussian_kernel = torch.exp(-(dst ** 2 / (2.0 * sigma ** 2)))
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    return gaussian_kernel


def finite_difference_kernel(kernel_size):
    """
    Finite difference kernel, something that looks like 
        [[-1, -1, -1],
         [ 0,  0,  0],
         [+1, +1, +1]]
    and it's transpose 
    Takes teh difference of the neighbours. x and y are in image axis 
    Args:
        kernel_size (int) : odd number, defines the size of the 
                neighbourhood of the pixel
        
    Returns:
        finite_x (torch.Tensor): shape (kernel_size, kernel_size)
        finite_y (torch.Tensor): shape (kernel_size, kernel_size)
    """
    assert kernel_size % 2, "`kernel_size` must be an odd number"

    a = torch.ones((kernel_size, kernel_size//2))
    b = torch.zeros((kernel_size, 1))
    c = -a.clone()

    finite_x = torch.cat([c, b, a], axis=1)
    finite_y =  finite_x.T

    return finite_x, finite_y

def finite_gaussian_kernel(kernel_size, sigma=0.5):
    """
    Returns kernels that are finite difference variant of
    the gaussian filter above.

    Args:
        kernel_size (int): odd number, defines the neighbourhood over which
                        the gaussian filter is applied
        sigma (float): default half a pixel, sigma defines how to weight pixels in the
                neighbourhood
 
    Returns:
        finite_x (torch.Tensor): shape (kernel_size, kernel_size)
        finite_y (torch.Tensor): shape (kernel_size, kernel_size)
    """
    assert kernel_size % 2, "`kernel_size` must be an odd number"
    
    finite_x, finite_y = finite_difference_kernel(kernel_size)
    g_kernel = gaussian_kernel(kernel_size, sigma)

    finite_x = finite_x * g_kernel
    finite_y = finite_y * g_kernel

    return finite_x, finite_y



