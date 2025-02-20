import numpy as np 
from skimage.metrics import structural_similarity as ssim
import esda.moran 
from scipy import stats
from scipy.spatial.distance import pdist, squareform 
from scipy.fft import fft2, fftfreq

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