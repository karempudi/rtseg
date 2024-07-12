import torch
import torch.nn.functional as F


def compute_wavelet_planes(anscombe_image, device: str = 'cpu'):
    """
    Args:
        anscombe_image:
        device:
    Returns: 
        wavelet_planes
    """
    # a hardcoded parameter in this method
    levels = 3

    H, W = anscombe_image.shape
    
    #print(f"Anscombe image shape: {H}x{W}")
    wavelet_planes = torch.zeros((levels+1, H, W), device=device)
    base_kernel1 = (1/16) * torch.tensor([1, 4, 6, 4, 1], device=device)
    base_kernel2 = (1/16) * torch.tensor([1, 0, 4, 0, 6, 0, 4, 0, 1], device=device)
    base_kernel3 = (1/16) * torch.tensor([1, 0, 0, 0, 4, 0, 0, 0, 6, 0, 0, 0, 4, 0, 0, 0, 1],
                                        device=device)
    
    kernel1 = (base_kernel1[:, None] * base_kernel1[None, :])[None, None, :]
    kernel2 = (base_kernel2[:, None] * base_kernel2[None, :])[None, None, :]
    kernel3 = (base_kernel3[:, None] * base_kernel3[None, :])[None, None, :]
    #kernel1.to(device)
    #kernel2.to(device)
    #kernel3.to(device)

    input_tensor = anscombe_image[None, None,:].to(device)
    last_tensor = torch.Tensor(input_tensor)
    
    out_tensor1 = F.conv2d(last_tensor, kernel1, padding='same')
    wavelet_planes[0] = (input_tensor - out_tensor1)[:, :]
    last_tensor = out_tensor1
    out_tensor2 = F.conv2d(last_tensor, kernel2, padding='same')
    wavelet_planes[1] = (last_tensor - out_tensor2)[:, :]
    last_tensor = out_tensor2
    out_tensor3 = F.conv2d(last_tensor, kernel3, padding='same')
    wavelet_planes[2] = (last_tensor - out_tensor3)[:, :]
    last_tensor = out_tensor3
    wavelet_planes[3] = last_tensor

    return wavelet_planes



def compute_spot_binary_mask(image, wavelet_plane, noise_threshold, device='cpu'):
    """
    Args:
        image:
        wavelet_plane_no:
        noise_threshold:
        device:

    Returns:
        A binary mask with 

    """
    #image = image.astype('float32')

    # first compute the wavelet planes

    #image_tensor = torch.from_numpy(image).to(device)
    # some transformation
    #anscombe_image = 2 * torch.sqrt(image_tensor + 3.0/8.0)
    #wavelet_planes = compute_wavelet_planes(anscombe_image, device=device)

    # filter wavelet planes

    # cleanup the binary spot mask for random noise.
    #wavelet_planes = None
    binary_spot_mask = None

    return binary_spot_mask


def detect_dots_wavelet(fluor_image, noise_threshold, segmask):
    """
    A function that outputs (x, y) coordinats of the dots in an image

    Args:
        fluor_image: fluorescence images that has the spots
        

    Returns:
        dots (np.ndarray): (N x 2) array of floats with coordinates
            of the dots from the origin of the image

    """

    # use wavelet method to get a binary spot mask

    # Do some additional computation to get a 
    # weighted centroid of the dots to finalize 
    # dot coordinates after some cleanup and split


    dots = None

    return dots

