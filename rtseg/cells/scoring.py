import numpy as np 
from skimage.metrics import structural_similarity as ssim
import esda.moran 
from scipy import stats
from scipy.spatial.distance import pdist, squareform 
from scipy.fft import fft2, fftfreq
import matplotlib.pyplot as plt


def score_all_fork_plots(fork_data, moran_weights):
#def score_all_fork_plots(heatmap_around_init, heatmap_around_init_pos_traps, 
 #                        flat_heatmap_init, flat_heatmap_init_around_init_pos_traps, 
  #                       moran_weights, e_dists):
    heatmap_init = fork_data['heatmap_around_init']
    heatmaps_init_trap = fork_data['heatmap_around_init_pos_traps']
    flat_heatmap_init = fork_data['flat_heatmap_init']
    flat_heatmaps_init_traps = fork_data['flat_heatmap_around_init_pos_traps']
    e_dists = fork_data['e_dists']

    nr_pos, nr_traps = heatmaps_init_trap.shape[-2:]

    corrcoeffs = np.full((nr_pos, nr_traps), np.nan)
    ssims = np.full((nr_pos, nr_traps), np.nan)
    morans = np.full((nr_pos, nr_traps, 3), np.nan)
    ks_results = np.full((nr_pos, nr_traps, 2), np.nan)
    sobolevs = np.full((nr_pos, nr_traps), np.nan)
    energies = np.full((nr_pos, nr_traps), np.nan)
    print(nr_pos, nr_traps)
    print(np.ndindex(nr_pos, nr_traps))
    for (i, j) in (np.ndindex(nr_pos, nr_traps)):
        flat_trap_heatmap = flat_heatmaps_init_traps[:,i,j]
        trap_heatmap = heatmaps_init_trap[:,:,i,j]

        if not np.any(flat_trap_heatmap):
            #Skips trap if there are no data points within the scoring window
            continue

        corrcoeffs[i,j] = score_correlation_coefficient(flat_heatmap_init, flat_trap_heatmap)

        ssims[i,j] = score_ssim(flat_heatmap_init, flat_trap_heatmap)

        morans[i,j,0], morans[i,j,1], morans[i,j,2] = score_cross_moran(flat_heatmap_init, flat_trap_heatmap, moran_weights)

        ks_results[i,j,0], ks_results[i,j,1] = score_kolmogorov_smirnov(flat_heatmap_init, flat_trap_heatmap)

        sobolevs[i,j] = score_sobolev_norm(heatmap_init, trap_heatmap)

        energies[i,j] = score_energy_test(flat_heatmap_init, flat_trap_heatmap, e_dists)

    all_scores = {
        'correlation': corrcoeffs,
        'ssim': ssims,
        'moran': morans,
        'ks': ks_results,
        'sobolev': sobolevs,
        'energies': energies
    }

    scores_median_mad = {}
    for key, values in all_scores.items():
        if values.ndim == 2:
            median_mad = calc_mad_around_median(values)
        else:
            #Deals with the cases where the metric gave multiple outputs 
            median_mad = calc_mad_around_median(values[:,:,0])

        scores_median_mad[key] = median_mad

    return all_scores, scores_median_mad  

def crop_single_trap_fork_plot(heatmap_trap, abins_inds_around_init, lbins_inds_around_init):
    
    heatmap_trap_init = heatmap_trap[np.ix_(abins_inds_around_init, lbins_inds_around_init)]

    return heatmap_trap_init

def score_correlation_coefficient(heatmap_init_pooled, heatmap_init_trap):
#Wants a flat heatmap
    corrcoeff = np.corrcoef(heatmap_init_pooled, heatmap_init_trap)[0, 1]

    return corrcoeff

def score_ssim(heatmap_init_pooled, heatmap_init_trap):
#Wants a flat heatmap
    max_val = np.max([np.max(heatmap_init_pooled), np.max(heatmap_init_trap)])
    min_val = np.min([np.min(heatmap_init_pooled), np.min(heatmap_init_trap)])

    ssim_val = ssim(heatmap_init_pooled, heatmap_init_trap, data_range=max_val-min_val)

    return ssim_val

def score_cross_moran(heatmap_init_pooled, heatmap_init_trap, weight):
#Wants flattened heatmap
    moran_bivar_heatmap = esda.moran.Moran_BV(heatmap_init_pooled, heatmap_init_trap, weight)
    cross_moran_I = moran_bivar_heatmap.I
    pval = moran_bivar_heatmap.p_sim
    z_score = moran_bivar_heatmap.z_sim

    return cross_moran_I, pval, z_score

def score_kolmogorov_smirnov(heatmap_init_pooled, heatmap_init_trap):
#Wants flattened heatmap
    ks_score, ks_pval = stats.ks_2samp(heatmap_init_pooled, heatmap_init_trap)

    return ks_score, ks_pval

def score_sobolev_norm(heatmap_init_pooled, heatmap_init_trap):

    diff_heatmap = heatmap_init_pooled - heatmap_init_trap
    fft_diff_heatmap = fft2(diff_heatmap)
    ny, nx = diff_heatmap.shape
    kx = fftfreq(nx)
    ky = fftfreq(ny)

    kx_2d, ky_2d = np.meshgrid(kx, ky)
    k_squared = kx_2d**2 + ky_2d**2
    k_squared[0,0] = 1

    weighted_sum_heatmaps = np.sum(np.abs(fft_diff_heatmap)**2 / (1+4*np.pi**2*k_squared))
    sobolev_norm = np.sqrt(weighted_sum_heatmaps)

    return sobolev_norm

def energy_dists(lbins_around_init, abins_around_init):

    xx, yy = np.meshgrid(lbins_around_init, abins_around_init)
    bin_centers = np.column_stack((xx.ravel(), yy.ravel()))
    dists = squareform(pdist(bin_centers))

    return dists

def score_energy_test(heatmap_init_pooled, heatmap_init_trap, dists):
#Wants flattened heatmaps
    n1, n2 = np.sum(heatmap_init_pooled) , np.sum(heatmap_init_trap)
    x1, x2, = heatmap_init_pooled/n1, heatmap_init_trap/n2

    phi_11 = np.sum(np.outer(x1, x1) * dists)
    phi_12 = np.sum(np.outer(x1, x2) * dists)
    phi_22 = np.sum(np.outer(x2, x2) * dists)
    energy_score = phi_11 + phi_22 - 2*phi_12

    return energy_score

def calc_mad_around_median(metric):

    medi = np.nanmedian(metric)
    mad = np.nanmedian(np.abs(metric-medi))
    below_medi = medi - mad
    above_medi = medi + mad
    
    return (medi, below_medi, above_medi) 

def score_plotter(scores, median_mad, plot_range, x_fill, stat_type):
    #Hook this to the window 
    flat_scores = scores.flatten()
    medi = median_mad[0]
    mad_below_med = median_mad[1]
    mad_above_med = median_mad[2]
    fill_area = [mad_below_med, mad_below_med, mad_above_med, mad_above_med]
    plt.figure()
    plt.plot(plot_range, flat_scores, 'o')
    plt.axhline(medi, linestyle='--', color='red')
    plt.fill(x_fill, fill_area, color='red', alpha=0.25, zorder=3)
    plt.xlabel('Channel')
    plt.ylabel(stat_type)
