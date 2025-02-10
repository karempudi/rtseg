
from rtseg.cells.utils import regionprops_custom, compute_arc_length, compute_projected_points
import numpy as np
import bisect
import sys

def compute_forkplot_stats(seg_mask, rotated_coords, position=0, timepoint=0,
            trap_locations=None, trap_width=40, traps_per_img=28):

    """
    Arguments:
        seg_mask: a labelled segmentation mask, we will rotate it to calculate internal 
                coordinates as backbone fitting is for cells growing horizontally in 
                the trap. In our imaging cells are growing vertically.
        rotated_coords: Coordinates on a 90 degree rotated image
        trap_locations: 
        position: 
        timepoint:

    Returns:
        a list with each element being a dictionary with properties for each dot.
    """

    data_for_forks = []
    if len(rotated_coords) == 0:
        return data_for_forks
    
    try:
        cell_mask_rot = np.rot90(seg_mask).copy()

        # calcuates poles, fitcoeff, arc_length for each cell using 
        # standard regionprops with some extra calucations
        props = regionprops_custom(cell_mask_rot)

        # assign dots to cells by label and calculate internal coordinates

        # grab the pixel in which dot lies
        dot_coords_int = rotated_coords.astype('int') 
        x, y = dot_coords_int[:, 0], dot_coords_int[:, 1]


        # for each dot we pick the cell label
        dot_labels = cell_mask_rot[x, y]

        unique_cell_labels, dot_counts = np.unique(dot_labels, return_counts=True)


        # figureing out trap number
        trap_locations_left = [loc-trap_width for loc in trap_locations]
        trap_locations_right = [loc+trap_width for loc in trap_locations]

        # iterate over unique cell labels, grab and compute appropriate things
        for i, single_cell_label in enumerate(unique_cell_labels, 0):
            if single_cell_label != 0: # 0 is for background label
                dot_idxs = np.where(dot_labels == single_cell_label)[0]
                # used as normalization 
                dots_per_cell = dot_counts[i]
                # grab the props of cell by indexing into the cell props array
                cell_prop = props[single_cell_label-1]
                left_index = bisect.bisect_left(trap_locations_left, cell_prop.centroid[0])
                right_index = bisect.bisect_right(trap_locations_right, cell_prop.centroid[0])
                if left_index - 1 == right_index:
                    trap_no = traps_per_img-right_index-1
                else:
                    trap_no = None
                fit_coeff = cell_prop.fit_coeff
                poles = cell_prop.poles
                #img = cell_prop.image
                #img_size = img.shape
                bbox = cell_prop.bbox
                arc_length = cell_prop.arc_length[0]
                #x_data = np.arange(-0.5, img_size[1]+0.5)
                #y_data = fit_coeff[0] * x_data**2 + fit_coeff[1] * x_data + fit_coeff[2]

                # for all dots inside the cell, compute internal coordinates

                #plt.figure()
                #plt.imshow(img)
                #plt.plot(x_data, y_data, 'r--')
                #plt.plot(poles[:, 0], poles[:, 1], '*')
                for dot_idx in dot_idxs:
                    dot_x, dot_y = rotated_coords[dot_idx]
                    local_x, local_y = dot_x - bbox[0], dot_y - bbox[1]
                    projected_point, internal_y = compute_projected_points(fit_coeff, np.array([[local_y, local_x]]))
                    distance_to_pole_along_arc = compute_arc_length(fit_coeff, poles[0, 0], projected_point[0, 0])

                    #plt.plot([local_y, projected_point[0, 0]], [local_x, projected_point[0, 1]], 'b--')
                    #plt.plot(local_y, local_x, 'go')
                    #plt.plot(projected_point[0, 0], projected_point[0, 1], 'b*')
                    #print(distance_to_pole_along_arc[0], arc_length)
                    dot_datapoint = {'position': position,
                                    'timepoint': timepoint,
                                    'trap': trap_no,
                                    'cell_label': single_cell_label,
                                    'area': cell_prop.area,
                                    'length': arc_length,
                                    'normalization_counts': dots_per_cell,
                                    'internal_coord': (distance_to_pole_along_arc[0], internal_y[0]),
                                    'normalized_internal_x': distance_to_pole_along_arc[0]/arc_length,
                                    'bbox': bbox,
                                    'global_coords': (dot_x, dot_y),
                                    'local_coords': (local_x, local_y),
                                    }
                    data_for_forks.append(dot_datapoint)
    
    except Exception as e:
        sys.stdout.write(f"Error {e} in computing fork plot stats Pos: {position}, timepoint: {timepoint}")
        sys.stdout.flush()
    finally:
        return data_for_forks
