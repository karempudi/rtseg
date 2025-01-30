
import numpy as np


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
    #threshold = np.quantile(heatmap, heatmap_threshold)
    #heatmap[heatmap > threshold] = 0.0

    heatmap[np.isnan(heatmap)] = 0.0

    y = (abins[0:-1] + abins[1:])/2
    x = lbins

    return heatmap, mean_cell_lengths, (x, y)


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