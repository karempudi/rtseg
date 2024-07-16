import torch
import torch.nn.functional as F
from skimage.measure import label
from skimage.segmentation import expand_labels

def clean_spots(image):
    """
    Clean isolated single pixels spots on the image
    """
    clean_kernel = torch.ones((1, 1, 3, 3)).to(image.device)
    conved = F.conv2d(image, clean_kernel, padding='same')
    image[conved == 1] = 0
    return image

def clean_hbreak(image):
    """
    Break two horizontal lines connected by one 1 pixel.
    """
    kernel1 = torch.tensor([[1.0, 1.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]])[None, None, :].to(image.device)
    kernel2 = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 0.0]])[None, None, :].to(image.device)
    conved1 = F.conv2d(image, kernel1, padding='same')
    conved2 = F.conv2d(image, kernel2, padding='same')
    result = torch.logical_and(conved1 == 7, conved2 == 2).float()

    find_ones = torch.ones((1, 1, 3, 3)).to(image.device)
    conved = F.conv2d(result, find_ones, padding = 'same')
    image[conved == 1] = 0
    return image

def clean_vbreak(image):
    """
    Break two vertical lines connected by 1 pixel
    """
    kernel1 = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [1.0, 0.0, 1.0]])[None, None, :].to(image.device)
    kernel2 = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])[None, None, :].to(image.device)
    conved1 = F.conv2d(image, kernel1, padding='same')
    conved2 = F.conv2d(image, kernel2, padding='same')
    result = torch.logical_and(conved1 == 7, conved2 == 2).float()

    find_ones = torch.ones((1, 1, 3, 3)).to(image.device)
    conved = F.conv2d(result, find_ones, padding = 'same')

    image[conved == 1] = 0
    return image

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



def compute_spot_binary_mask(image, seg_mask, wavelet_plane_no = 2,
             noise_threshold = 3.0, noise_level_division=0.7, device='cpu'):
    """
    Args:
        image:
        wavelet_plane_no:
        noise_threshold:
        device:

    Returns:
        A binary mask with 

    """
    image = image.astype('float32')

    # first compute the wavelet planes

    image_tensor = torch.from_numpy(image).to(device)
    # some transformation
    anscombe_image = 2 * torch.sqrt(image_tensor + 3.0/8.0)
    # waveletplanes
    w = compute_wavelet_planes(anscombe_image, device=device)

    # filter wavelet planes
    w = w.to(device)
    seg_mask = torch.from_numpy(seg_mask)
    seg_mask = seg_mask.to(device)
    ind = seg_mask > 0

    w_masked = w[:, ind]
    w_mean = torch.mean(w_masked, dim=1)[:, None]
    noise_level = torch.median(torch.abs(w_masked - w_mean), dim=1).values / noise_level_division
    threshold = noise_threshold * noise_level
    
    filtered_image = torch.zeros_like(w[wavelet_plane_no], device=device)

    filtered_image[torch.abs(w[wavelet_plane_no]) >= threshold[wavelet_plane_no]] = 1

    filtered_image *= w[wavelet_plane_no]

    binary_mask = torch.zeros_like(filtered_image, device=device)
    binary_mask[filtered_image > 0] = 1
    

    # cleanup the binary spot mask for random noise.

    cleaned = clean_spots(binary_mask[None, None, :])
    cleaned = clean_hbreak(cleaned)
    cleaned = clean_vbreak(cleaned)
    cleaned = clean_spots(cleaned)
    # thicken using skimage
    
    cleaned_mask = cleaned[0][0].cpu().numpy()

    binary_spot_mask = expand_labels(label(cleaned_mask)) > 0

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

