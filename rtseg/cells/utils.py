
import numpy as np
import cv2
import edt # type: ignore
import math
from numpy.linalg import lstsq
import matplotlib.pyplot as plt
from skimage.measure import regionprops

def compute_boundary_pixels(image, distance_threshold: int = 3):
    """
    Args:
        image (np.ndarray): a labelled slice of the image
    Returns:
        (y_boundary, x_boundary): indices of the pixels on the boundary
    """
    assert distance_threshold % 2 == 1, "`For now distance threshold is set to be odd number"
    dilation_radius = math.floor(distance_threshold / 2)
    padded_image = np.pad(image, (dilation_radius + 1, dilation_radius + 1)).astype('uint8')

    # to get boundary pixels, perform edt == 1 on the non-dilated image, padded image
    boundary_image = (edt.edt(padded_image) == 1)
    boundary_image = boundary_image[dilation_radius+1: -dilation_radius-1,
                                dilation_radius+1: -dilation_radius-1]
    y_boundary, x_boundary = np.nonzero(boundary_image)
    return y_boundary, x_boundary

def compute_backbone_coordinates(fit_coeff, img_size):
    """
    Computes backbone pixel coordiantes by looking at the pixels
    through with the qudartic curve passes.
    Args:
        fit_coeff:
        img_size 
    Returns:
        fit_coord: N x 2 numpy array with the backbone pixels coordinates
    """
    x_data = np.arange(-0.5, img_size[1]+0.5)
    y_data = fit_coeff[0] * x_data**2 + fit_coeff[1] * x_data + fit_coeff[2]
    if abs(fit_coeff[0]) < 1e-4 and round(y_data[0]) == round(y_data[-1]):
        fit_coord = np.vstack((np.arange(0, img_size[1]), round(y_data[0]) * np.ones((img_size[1],)))).T
        fit_coord = fit_coord.astype('int')
    else:
        y_data_round = np.round(y_data)
        stat_pt = -fit_coeff[1] / (2 * fit_coeff[0]) #  ax^2 + bx + c = 0 --> -b/2a is the point where derivative==0
        if stat_pt > -0.5 and stat_pt < img_size[1] - 1:
            round_stat_pt_val = round(fit_coeff[0] * (stat_pt**2) + fit_coeff[1] * stat_pt + fit_coeff[2])
            round_stat_pt = round(stat_pt)
            upp_bound = np.zeros((img_size[1]),) # upper bound defined for each pixel on the x-axis
            low_bound = np.zeros((img_size[1]),) # lower bound defined for each pixel on the x-axis
        
            if fit_coeff[0] > 0:
                upp_bound[0:round_stat_pt] = y_data_round[0:round_stat_pt]
                upp_bound[round_stat_pt] = max(y_data_round[round_stat_pt:round_stat_pt+2])
                upp_bound[round_stat_pt+1:] = y_data_round[round_stat_pt+2:]
                low_bound[0:round_stat_pt] = y_data_round[1:round_stat_pt+1]
                round_down_point = np.argwhere(y_data_round[1:] - y_data[1:] == 0.5)
                low_bound[round_down_point] = low_bound[round_down_point]-1 
                low_bound[round_stat_pt] = round_stat_pt_val 
                low_bound[round_stat_pt+1:] =  y_data_round[round_stat_pt+1:-1]
                                            
            else:
                upp_bound[0:round_stat_pt] = y_data_round[1:round_stat_pt+1]
                upp_bound[round_stat_pt] = round_stat_pt_val
                upp_bound[round_stat_pt+1:] = y_data_round[round_stat_pt+1:-1]
                low_bound[0:round_stat_pt] =  y_data_round[0:round_stat_pt]
                low_bound[round_stat_pt] = max(y_data_round[round_stat_pt:round_stat_pt+2])
                low_bound[round_stat_pt+1:] = y_data_round[round_stat_pt+2:]
        else:
            if (fit_coeff[0] > 0 and stat_pt > img_size[1]-1) or (fit_coeff[0] < 0 and stat_pt < -0.5):
                upp_bound = y_data_round[:-1]
                low_bound = y_data_round[1:]
                round_down_point = np.argwhere(low_bound - y_data[1:] == 0.5)
                low_bound[round_down_point] = low_bound[round_down_point]-1
            else:
                upp_bound = y_data_round[1:]
                low_bound = y_data_round[:-1]
            
        
        # clip the upper bound and lower bounds
        # and populate the pixels
        low_bound = np.maximum(low_bound, 0).astype('int')
        upp_bound = np.minimum(upp_bound, img_size[0]-1).astype('int')
        n_pixels_per_col = upp_bound - low_bound + 1
        n_pixels_per_col[n_pixels_per_col < 1] = 0
        fit_coord = np.zeros((int(np.sum(n_pixels_per_col)), 2))
        idx = 0
        for i in range(len(n_pixels_per_col)):
            fit_coord[idx:idx+n_pixels_per_col[i], 0] = i
            fit_coord[idx:idx+n_pixels_per_col[i], 1] = np.arange(low_bound[i], upp_bound[i]+1)
            idx += n_pixels_per_col[i]
        
        fit_coord = fit_coord.astype('int')
    
    return fit_coord

def compute_arc_length(fit_coeff, first_point, last_point):
    """
    Computes the arc length between the first point and last point
    on the quadradtic curve used to fit the backbone. For cell length
    the first and last point will be the poles of the cells.
    For internal x coordinate, you will use one pole.

    Args:
        fit_coeff: 3 values corresponding to the coefficients of 
            the quadratic used to fit the backbone
        first_point: x,coordinate of the first point
        last_point: x coordinate of the last point
    
    Returns:
        arc_length: float

    """
    x_0 = first_point
    x_end = last_point
    if abs(fit_coeff[0]) > 1e-10:
        transf_points = (fit_coeff[0] / abs(fit_coeff[0])) * np.array([2*fit_coeff[0]*x_0 + fit_coeff[1] , 2*fit_coeff[0]*x_end + fit_coeff[1]])
        sqrt_exp = np.sqrt(1.0 + transf_points**2)
        arc_length = abs(np.diff(transf_points * sqrt_exp + np.log(transf_points + sqrt_exp)) / (4 * abs(fit_coeff[0])))
    else:
        fit_points = [fit_coeff[1] * x_0 + fit_coeff[2], fit_coeff[1] * x_end + fit_coeff[2]]
        arc_length = np.array([math.sqrt((x_0 - x_end) ** 2 + (fit_points[0] - fit_points[1]) ** 2)])

    return arc_length

def compute_fit_coeff_poles_arc_length(image, distance_threshold: int=3,
                weight_quadratic:int = 3, plot:bool = False):
    """

    Args:
        image (np.ndarray): a labelled slice of the image
        distance_threshold (int): defines the dilation done using a disk / ellipse
            structuring element after padding the image
        weight_quadratic (int) : power for the weight when computing least square solutions
            during the qudartic fit
        
        plot (bool): default False, if true will plot the cell with the fit
        
    Returns: a tuple (fit_coeff, poles, arc_length, boundary_pixels)
        fit_coeff (np.ndarray) of shape (3,). The coefficients
            of the quadtartic equation that is fit for the 
            backbone of the cell.
    """
    assert distance_threshold % 2 == 1, "`For now distance threshold is set to be odd number"
    dilation_radius = math.floor(distance_threshold / 2)
    padded_image = np.pad(image, (dilation_radius + 1, dilation_radius + 1)).astype('uint8')

    # to get boundary pixels, perform edt == 1 on the non-dilated image, padded image
    boundary_image = (edt.edt(padded_image) == 1)
    boundary_image = boundary_image[dilation_radius+1: -dilation_radius-1,
                                dilation_radius+1: -dilation_radius-1]
    boundary_pixels = np.nonzero(boundary_image)

    structuring_radius = distance_threshold # careful with this detail
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (structuring_radius, structuring_radius))
    
    dilated_image = cv2.dilate(padded_image, se)
    region_edt = edt.edt(dilated_image)
    label_dist = region_edt * (region_edt >= distance_threshold)
    label_dist = label_dist[dilation_radius+1: -dilation_radius-1,
                            dilation_radius+1: -dilation_radius-1]

    # Do the quadratic fit and calculate the fit_coeff
    y_coords, x_coords = np.nonzero(label_dist)
    pixel_values = label_dist[y_coords, x_coords]
    weights = np.sqrt(pixel_values ** weight_quadratic)
    A = np.vstack((x_coords**2, x_coords, np.ones(x_coords.shape)))
    w_rep = np.tile(weights, [3, 1])
    wA = (w_rep * A).T
    wB = weights * y_coords

    fit_coeff, _, _, _ = lstsq(wA, wB, rcond=-1)


    # computing backbone pixels so that we can get the poles
    fit_coord = compute_backbone_coordinates(fit_coeff, image.shape)
    fit_im = np.zeros(image.shape, dtype='bool')
    fit_im[fit_coord[:, 1], fit_coord[:, 0]] = True
    backbone_im = image * fit_im
    y_backbone, x_backbone = np.nonzero(backbone_im)

    # poles
    poles = np.zeros((2, 2))
    x_0, x_end = min(x_backbone) - 0.5, max(x_backbone) + 0.5 # deviation from matlab code as np.nonzero doesn't sort in x dimension
    poles[:, 0] = [x_0, x_end]
    poles[:, 1] = fit_coeff[0] * poles[:, 0]**2 + fit_coeff[1] * poles[:, 0] + fit_coeff[2]

    # computing arc length
    arc_length = compute_arc_length(fit_coeff, x_0, x_end)

    if plot:
        x_data = np.arange(-0.5, image.shape[1]+0.5)
        y_data = fit_coeff[0] * x_data**2 + fit_coeff[1] * x_data + fit_coeff[2]
        plt.figure()
        plt.imshow(image)
        plt.plot(poles[:, 0], poles[:, 1], '*')
        plt.plot(x_data, y_data, 'r--')
        plt.show()

    return (fit_coeff, arc_length, poles, boundary_pixels)


def compute_projected_points(fit_coeff, dot_coordinates):
    """
    Compute the projected points for each of the dot coordinates
    in the (N x 2) array and return (N x 2) projected points
    onto the normal, and distances of the dots to the 
    backbone, which are the internal y_coordinate

    Dot coordiantes are ofcourse related to the cell image origin, 
    and not from the image origin.

    Args:
        fit_coeff: 3 values corresponding to the coefficients
            of the quadratic of the backbone
        dot_coordinates (np.ndarray): (N x 2) floats
    
    Returns:
        projected_points (np.ndarray): (N x 2) floats
        interna_y (np.ndarray): (N) floats
    """
    assert dot_coordinates.shape[0] > 0, "`Not enough dots to compute internal coordinates`"
    projected_points = np.zeros_like(dot_coordinates)
    internal_y = np.zeros((dot_coordinates.shape[0],))
    p0, p1, p2 = fit_coeff
    for i in range(dot_coordinates.shape[0]):
        x0, y0 = dot_coordinates[i]
        if abs(p0) > 1e-10:
            x_star = np.roots([2*p0**2, 3*p0*p1, 2*p0*(p2-y0) + (p1**2+1), p1*(p2-y0)-x0])
            x_star[abs(np.imag(x_star)) < 1e-10] = np.real(x_star[abs(np.imag(x_star)) < 1e-10])
            x_star = np.real(x_star[np.isreal(x_star)])
            x_mid = -p1/(2*p0)
            if x0 > x_mid:
                projected_point_x = np.max(x_star)
            else:
                projected_point_x = min(x_star)
        else:
            projected_point_x = x0 - ((p1 * (p2 - p1)) / ((p1**2 + 1)))
        
        projected_point_y =  p0 * projected_point_x ** 2 + p1 * projected_point_x + p2
        projected_points[i][0] = projected_point_x
        projected_points[i][1] = projected_point_y

        internal_y[i]  = math.sqrt((projected_point_x- x0)**2 + (projected_point_y - y0)**2)
        # Sign inversion depending on which side of the backbone the 
        # dot lies on
        if projected_point_y < y0:
            internal_y[i] = -internal_y[i]

    return projected_points, internal_y


def regionprops_custom(label_img):
    """
    Custom region props function that calculates additional properties
    such as backbone pixels, arc length, poles of the cell. It also
    stores the coefficients of the backbone quadratic equation fit.

    Args:
        label_img (np.ndarray) : labelled image numpy array.
    
    Returns:
        regionprops (list): a list of regionprops with properties for each
            cell. Uses the skimage.measure.regionprops and add on additional
            property calculations
    Note:
        Additional properites added to each region properties.
            1. fit_coeff: three floating point numbers, the coefficients 
                (a, b, c)  with we solve ax^2 + bx + c = 0 type equation
                for each cell fitting a quadratic using weighted least squares
            2. arc_length: length of the cell
            3. poles: Each cell has two poles, numpy array of shape (2, 2).
                [[x1, y1], [x2, y2]]. We are using image coordinates and not
                row colums.
            4. boundary_pixels
    """
    props = regionprops(label_img, extra_properties=(compute_fit_coeff_poles_arc_length,))
    for cellprop in props:
        #print(f"Cell no: {cellprop.label}")
        cellprop.fit_coeff, cellprop.arc_length, cellprop.poles, cellprop.boundary_pixels = cellprop.compute_fit_coeff_poles_arc_length

    return props


def compute_dots_inside_cell():


    # Convert global coordinate to local coordinate 
    # and then compute internal coordiantes
    return None, None, None, None