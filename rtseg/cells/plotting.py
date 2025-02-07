
import numpy as np
from scipy.special import erf
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings("ignore", message="Mean of empty slice")


def generate_fork_plot(areas, lengths, longs, counts, 
                bin_scale=20, pixel_size=0.046, heatmap_threshold=0.99,
                abins=None, lbins=None):
    
    np.seterr(divide='ignore', invalid='ignore')
    areas = areas * (pixel_size **2)
    lengths = lengths * pixel_size

    longs = longs - 0.5

    if abins is None and lbins is None:
        amin = np.quantile(areas, 0.005)
        amax = np.quantile(areas, 0.98)

        ainds = np.where(np.logical_and(areas >= amin, areas <= amax))[0]
        rescaled_lengths = lengths * longs
        rescaled_lengths_filtered = rescaled_lengths[ainds]

        lmax = max(rescaled_lengths_filtered)
        lmin = min(rescaled_lengths_filtered)
        lminmax = max(lmax, -lmin)
        lnrbins = round(2 * lminmax * bin_scale)
        if lnrbins % 2 != 0:
            lnrbins += 1
    
        lbins = np.linspace(-lminmax, lminmax, lnrbins)
        anrbins = lnrbins
        abins = np.linspace(amin, amax, anrbins)
    
    else:
        lnrbins = len(lbins)
        anrbins = lnrbins

    heatmap = np.zeros(shape=(anrbins-1, lnrbins-1))
    mean_cell_lengths = np.zeros((anrbins-1,))
    for i in range(0, anrbins-1):
        sel_dots = np.logical_and(areas > abins[i], areas <= abins[i+1])
        sel_lengths = lengths[sel_dots]
        sel_counts = counts[sel_dots]
        sel_long = longs[sel_dots]
        h1,_ = np.histogram(sel_long * sel_lengths, lbins)
        normFactor = np.sum(1./sel_counts)
        heatmap[i, :] = h1 / normFactor
        mean_cell_lengths[i] = np.mean(sel_lengths)

    # threshold the heatmap
    thresh = np.quantile(heatmap, heatmap_threshold)
    heatmap[heatmap > thresh] = thresh

    heatmap[np.isnan(heatmap)] = 0.0

    y = (abins[0:-1] + abins[1:])/2
    x = lbins

    return heatmap, mean_cell_lengths, abins, lbins, (x, y)


def generate_abins_lbins(areas, lengths, longs, pixel_size=0.046, bin_scale=20):
    areas = areas * (pixel_size **2)
    lengths = lengths * pixel_size
    
    amin = np.quantile(areas, 0.005)
    amax = np.quantile(areas, 0.98)

    ainds = np.where(np.logical_and(areas >= amin, areas <= amax))[0]
    longs = longs - 0.5
    rescaled_lengths = lengths * longs
    rescaled_lengths_filtered = rescaled_lengths[ainds]

    lmax = max(rescaled_lengths_filtered)
    lmin = min(rescaled_lengths_filtered)
    lminmax = max(lmax, -lmin)
    lnrbins = round(2 * lminmax * bin_scale)
    if lnrbins % 2 != 0:
        lnrbins += 1
    
    lbins = np.linspace(-lminmax, lminmax, lnrbins)
    anrbins = lnrbins
    abins = np.linspace(amin, amax, anrbins)

    return abins, lbins

def get_bulk_init_area(areas, counts, longs, lengths, min_length=0.44, pixel_size=0.046, arb_div_area = 3):
#The arb_div_area variable is a guess of the approximate division area of cells in µm^2. The default is 3, but it can be changed 

    start_point = np.array([2, 2, 0.2])

    dots_per_cell, area_bins = get_dots_per_cell(areas, counts, longs, lengths, min_length=min_length, pixel_size=pixel_size, dots_per_cell_bin_scale=50)

    nan_dot_inds = ~np.isnan(dots_per_cell)
    area_bins = area_bins[nan_dot_inds]
    dots_per_cell = dots_per_cell[nan_dot_inds]

    sel_area_bins = area_bins < arb_div_area
    area_bins_fit = area_bins[sel_area_bins]
    dots_per_cell_fit = dots_per_cell[sel_area_bins]

    the_fit, _ = curve_fit(dpc_fit, area_bins_fit, dots_per_cell_fit, p0=start_point)

    #Plotting code that can be used for debugging  
    #plt.plot(area_bins, dots_per_cell, 'bs')
    #plt.plot(area_bins_fit, dpc_fit(area_bins_fit, *the_fit), 'r')
    #plt.show()
    init_area = the_fit[1]

    return init_area


def get_dots_per_cell(areas, counts, longs, lengths, min_length=0.44, pixel_size=0.046, dots_per_cell_bin_scale=50):
    
    areas = areas * pixel_size**2
    amin = np.quantile(areas, 0.005)
    amax = np.quantile(areas, 0.98)
    anrbins = round((amax-amin) * dots_per_cell_bin_scale)
    abins = np.linspace(amin, amax, anrbins)
    
    lengths = lengths * pixel_size
    longs = longs - 0.5
    absLongsLengths = np.abs(longs * lengths)

    dots_per_cell = np.zeros((anrbins-1,))
    for i in range(0, anrbins-1):
        sel_dots = np.logical_and(areas > abins[i], areas <= abins[i+1])
        sel_longs = absLongsLengths > min_length
        sel_counts = counts[sel_dots]
        norm_factor = sum(1/sel_counts)
        dots_per_cell[i] = np.sum(sel_dots & sel_longs)/norm_factor

    area_bins = (abins[0:-1] + abins[1:])/2
    
    return dots_per_cell, area_bins

def dpc_fit(x, b, x0, s):
    return b * (1 + erf((x - x0) / (np.sqrt(2) * s)))

def slice_fork_plot_around_init(abins, lbins, heatmap, mean_cell_lengths, init_area, init_area_cv=0.11):
#Slices out the fork plot around the area of initiation. Intended to be used for the full fork plot. 
#It helps reducing computational time further downstream and it makes sure that we look at what is interesting
#and not just features that otherwise are the same in all fork plots.
    init_area_low = init_area - init_area_cv*init_area
    init_area_high = init_area + init_area_cv*init_area

    abins_inds_around_init = np.where(np.logical_and(init_area_low <= abins, abins <= init_area_high))
    abins_inds_around_init = abins_inds_around_init[0]
    abins_inds_around_init = np.append(abins_inds_around_init, abins_inds_around_init[-1]+1)
    area_bins_around_init = abins[abins_inds_around_init]
    mean_cell_lengths_around_init = mean_cell_lengths[abins_inds_around_init[0:-1]]
    max_cell_length_around_init = np.max(mean_cell_lengths_around_init)
    lbins_inds_around_init = np.where(np.logical_and(-0.5*max_cell_length_around_init <= lbins , lbins <= 0.5*max_cell_length_around_init))
    lbins_inds_around_init = lbins_inds_around_init[0]
    lbins_inds_around_init = np.insert(lbins_inds_around_init, 0, [lbins_inds_around_init[0]-2, lbins_inds_around_init[0]-1])
    lbins_inds_around_init = np.append(lbins_inds_around_init, [lbins_inds_around_init[-1]+1, lbins_inds_around_init[-1]+2])
    lbins_around_init = lbins[lbins_inds_around_init]

    heatmap_around_init = heatmap[np.ix_(abins_inds_around_init, lbins_inds_around_init)]

    return area_bins_around_init, lbins_around_init, heatmap_around_init, mean_cell_lengths_around_init, abins_inds_around_init