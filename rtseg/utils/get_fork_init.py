import numpy as np 
from pathlib import Path
from rtseg.cells.plotting import fast_generate_fork_plot, get_bulk_init_area, slice_fork_plot_around_init
import polars as pl 
import pandas as pd 
import glob 
import pysal.lib
from rtseg.cells.scoring import energy_dists
import sys

def read_all_fork_data_around_init(param):
    
    try:
        save_dir = Path(param.Save.directory)
        use_pyarrow = False 

        if param.Forkplots.polars is False:
            forks_filenames = [position_dir / Path('forks.csv') for position_dir in list(save_dir.glob('Pos*'))]
            dataframes = [pd.read_csv(filename) for filename in forks_filenames]
            data = pd.concat(dataframes, ignore_index=True)
        else:
            # read parquet files
            fork_filenames = glob.glob(str(save_dir / Path('Pos[0-9]*/forks.parquet')))
            columns_to_extract = ['area', 'length', 'normalized_internal_x', 'normalization_counts', 'position', 'trap']
            data = pl.read_parquet(fork_filenames, use_pyarrow=use_pyarrow, columns=columns_to_extract)

        areas = data['area'].to_numpy()
        lengths = data['length'].to_numpy()
        longs = data['normalized_internal_x'].to_numpy()
        counts = data['normalization_counts'].to_numpy()

        bin_scale = param.Forkplots.bin_scale
        heatmap_threshold = param.Forkplots.heatmap_threshold
        pixel_size = param.Forkplots.pixel_size
        min_length = param.Forkplots.min_length
        arb_div_area = param.Forkplots.arb_div_area
        init_area_cv = param.Forkplots.init_area_cv 
      
        heatmap, mean_cell_lengths, abins, lbins, _ = fast_generate_fork_plot(areas, lengths, longs, counts,
                        bin_scale=bin_scale, pixel_size=pixel_size, heatmap_threshold=heatmap_threshold)
        
        init_area = get_bulk_init_area(areas, counts, longs, lengths, min_length=min_length, 
                                           pixel_size=pixel_size, arb_div_area=arb_div_area)
            
        area_bins_around_init, lbins_around_init, heatmap_around_init, mean_cell_lengths_around_init, abins_inds_around_init, lbins_inds_around_init = slice_fork_plot_around_init(abins, lbins, heatmap, mean_cell_lengths, init_area, init_area_cv)

        #Number of dots around initiation
        amin_init = np.min(area_bins_around_init)
        amax_init = np.max(area_bins_around_init)
        lmin_init = np.min(lbins_around_init)
        lmax_init = np.max(lbins_around_init)
        ar_pix = areas * pixel_size**2
        le_pix = lengths * pixel_size
        lo_cent = longs - 0.5
        len_long_rescaled = le_pix * lo_cent
        inds_around_init = np.where(np.logical_and(np.logical_and(ar_pix >= amin_init, ar_pix <= amax_init), 
                                                   np.logical_and(len_long_rescaled >= lmin_init, 
                                                   len_long_rescaled <= lmax_init)))[0]
        lo_init = lo_cent[inds_around_init]
        nr_dots_init = lo_init.size

         #Flat heatmap
        flat_heatmap_init = heatmap_around_init.flatten()

        #Calculation for Moran's I
        moran_weight = pysal.lib.weights.lat2W(heatmap_around_init.shape[0], heatmap_around_init.shape[-1], rook=True)

        #Calculation for energy test
        e_dists = energy_dists(lbins_around_init, area_bins_around_init)

        all_pos_numbers = data['position'].to_numpy()
        pos_inds = np.unique(all_pos_numbers)
        nr_pos = len(pos_inds)
        all_trap_numbers = data['trap'].to_numpy()
        nr_traps = param.BarcodeAndChannels.num_blocks_per_image*param.BarcodeAndChannels.num_traps_per_block
        trap_nrs = np.arange(nr_traps)

        heatmap_around_init_dims = heatmap_around_init.shape
        heatmap_around_init_pos_traps = np.zeros(shape=(heatmap_around_init_dims[0], heatmap_around_init_dims[1], nr_pos, nr_traps))
        flat_heatmap_around_init_pos_traps = np.zeros(shape=(heatmap_around_init_dims[0]*heatmap_around_init_dims[1], nr_pos, nr_traps))
        all_traps_nr_dots = np.zeros(shape=(nr_pos, nr_traps))

        for (i, j) in (np.ndindex(nr_pos, nr_traps)):


            pos = pos_inds[i]
            trap = trap_nrs[j] 

            # check if the trap parquet exists
            filename = Path(param.Save.directory) / Path('Pos' + str(pos)) / Path('forks.parquet/trap=' + str(trap))
            if not filename.exists():
                continue

            trap_inds = np.where((all_pos_numbers == pos) & (all_trap_numbers == trap))
            areas_trap = areas[trap_inds]
            longs_trap = longs[trap_inds]
            lengths_trap = lengths[trap_inds]
            counts_trap = counts[trap_inds] 

            heatmap_trap, _, _, _, _ = fast_generate_fork_plot(areas_trap, lengths_trap, longs_trap, 
                                                               counts_trap, abins=abins, lbins=lbins,
                                                               bin_scale=bin_scale, pixel_size=pixel_size,
                                                               heatmap_threshold=heatmap_threshold)
            
            heatmap_trap_init = heatmap_trap[np.ix_(abins_inds_around_init, lbins_inds_around_init)]
            flat_heatmap_trap_init = heatmap_trap_init.flatten()
            heatmap_around_init_pos_traps[:,:,i,j] = heatmap_trap_init
            flat_heatmap_around_init_pos_traps[:,i,j] = flat_heatmap_trap_init
            
            ar_pix_trap = areas_trap * pixel_size**2
            le_pix_trap = lengths_trap * pixel_size
            lo_cent_trap = longs_trap - 0.5
            len_long_rescaled_trap = le_pix_trap * lo_cent_trap
            inds_around_init_trap = np.where((ar_pix_trap >= amin_init) & (ar_pix_trap <= amax_init) & 
                                                   (len_long_rescaled_trap >= lmin_init) & 
                                                   (len_long_rescaled_trap <= lmax_init))[0]
            lo_init_trap = lo_cent_trap[inds_around_init_trap]
            nr_dots_init_trap = lo_init_trap.size
            all_traps_nr_dots[i,j] = nr_dots_init_trap

        return {

            'heatmap_around_init': heatmap_around_init,
            'heatmap_around_init_pos_traps': heatmap_around_init_pos_traps,
            'flat_heatmap_init': flat_heatmap_init,
            'flat_heatmap_around_init_pos_traps': flat_heatmap_around_init_pos_traps, 
            'init_area': init_area, 
            'e_dists': e_dists,
            'nr_dots_init': nr_dots_init,
            'all_Traps_nr_dots': nr_dots_init_trap

        }, moran_weight
        
    except Exception as e:
        sys.stdout.write(f"Reading data failed due to {e} ...\n")
        sys.stdout.flush()