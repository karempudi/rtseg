
import numpy as np
from rtseg.dotdetection.detect import compute_spot_binary_mask
from skimage.measure import label, regionprops
#from rtseg.cells.utils import regionprops_custom

def compute_dot_coordinates(fluor_img, cell_mask, param):
    """
    Wrapper around detecing dots and computing internal coordinates

    """    
    raw_shape = fluor_img.shape
    fluor_img_rot = np.rot90(fluor_img).copy()
    cell_mask_rot = np.rot90(cell_mask).copy()

    dot_params = param.Dots

    binary_spot_mask = compute_spot_binary_mask(fluor_img_rot, cell_mask_rot,
                noise_threshold=dot_params.noise_threshold,
                wavelet_plane_no=dot_params.wavelet_plane_no,
                device=param.Hardware.device)

    spot_stats = regionprops(label(binary_spot_mask), fluor_img_rot)
    min_spot_area = dot_params.min_spot_area
    #max_axes_ratio = dot_params.max_axes_ratio


    spot_filtered = [item for item in spot_stats if item.area > min_spot_area]
    #TODO do spots filtering well
    #spot_axes_ratio = np.array([spot.axis_major_length / spot.axis_minor_length for spot in spot_filtered])
    #spot_areas = np.array([spot.area for spot in spot_filtered])
    ## indices in spot filtered based on area
    dot_coords = [spot.centroid_weighted for spot in spot_filtered]

    dot_coords_np = np.array(dot_coords)

    dots_on_raw_img = np.zeros_like(dot_coords_np)

    # tranforming to original image
    dots_on_raw_img[:, 0] = dot_coords_np[:, 1]
    dots_on_raw_img[:, 1] = -dot_coords_np[:, 0] + raw_shape[1] - 1

    return dots_on_raw_img


