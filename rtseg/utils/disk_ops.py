
import numpy as np
import sys
from pathlib import Path
from skimage.io import imsave, imread
import zarr
from numcodecs import Zlib
from rtseg.utils.db_ops import read_from_db
from rtseg.cells.plotting import generate_fork_plot, generate_abins_lbins, get_bulk_init_area, slice_fork_plot_around_init
from rtseg.cells.plotting import fast_generate_fork_plot
from tifffile import imwrite # used for finer compression things
import h5py
import polars as pl
import pandas as pd
import glob
import os
import pysal.lib
from rtseg.cells.scoring import energy_dists

def write_files(event_data, event_type, param):
    """
    Function that write data of all kinds in appropirate directories
    based on event_type

    Arguments:
        event_data: dict containing data to write
            event data should always have 'position' and 'timepoint' keys

        event_type: 'phase', 'fluor', 'segmented_cells_by_trap'

        param: param used from param file, contains experimental information
        
    """

    try:
        if 'position' not in event_data.keys():
            raise KeyError
        if 'timepoint' not in event_data.keys():
            raise KeyError

        save_dir = Path(param.Save.directory)
        position_dir = save_dir / Path('Pos' + str(event_data['position']))

        # create position directory
        if not position_dir.exists():
            position_dir.mkdir(exist_ok=True, parents=True)

        
        if event_type == 'phase':
            # create events_dir
            events_dir = position_dir / Path(event_type)
            if not events_dir.exists():
                events_dir.mkdir(exist_ok=True, parents=True)
            
            # put image in phase directory at position with timepoint
            image_filename = 'phase_' + str(event_data['timepoint']).zfill(4) + '.tiff'
            image_filename = events_dir / Path(image_filename)
            imsave(image_filename, event_data['image'].astype('uint16'))

        elif event_type == 'fluor':
            # create events_dir
            events_dir = position_dir / Path(event_type)
            if not events_dir.exists():
                events_dir.mkdir(exist_ok=True, parents=True)
            
            # put image in fluor directory 
            image_filename = 'fluor_'  + str(event_data['timepoint']).zfill(4) + '.tiff'
            image_filename = events_dir / Path(image_filename)
            imsave(image_filename, event_data['image'].astype('uint16'), check_contrast=False)
        
        elif event_type == 'seg_mask':
            # create events dir
            events_dir = position_dir / Path(event_type)
            if not events_dir.exists():
                events_dir.mkdir(exist_ok=True, parents=True)

            # put image in the seg mask directory
            image_filename = 'mask_' + str(event_data['timepoint']).zfill(4) + '.tiff'
            image_filename = events_dir / Path(image_filename)
            imwrite(image_filename, event_data['image'], compression='zlib', compressionargs={'level': 6})
        
        elif event_type == 'segmented_cells_by_trap':
            cells_filename = position_dir / Path('cells.zarr')
            compressor = Zlib(level=6) # change compression level if you don't like the time it takes
            # TODO : set height and width later
            height, _ = event_data['image'].shape
            num_blocks = param.BarcodeAndChannels.num_blocks_per_image
            num_traps_per_block = param.BarcodeAndChannels.num_traps_per_block
            trap_width = param.Save.trap_width
            image = np.zeros((height, param.Save.trap_width * num_blocks * num_traps_per_block))
            _, width = image.shape
            if len(event_data['trap_locations_list']) == num_blocks * num_traps_per_block:
                # loop over traps and crop and construct image
                for i, loc in enumerate(event_data['trap_locations_list']):
                    image[:, i * trap_width: (i+1) * trap_width] = event_data['image'][:, loc-(trap_width//2): loc+(trap_width//2)]


            if event_data['timepoint'] == 0:
                # intialize  a zarr to write into and then set the slices correctly            
                cells_array = zarr.convenience.open(cells_filename, mode='a', shape=(1, height, width),
                                chunks=(1, height, param.Save.trap_width), order='C', 
                                dtype='uint8', compressor=compressor)
                cells_array[0] = image # image will be a chopped channels image stitched together
                
            else:
                # write into the zarr directly
                cells_array = zarr.convenience.open(cells_filename, mode='a', shape=(1, height, width),
                                chunks=(1, height, param.Save.trap_width), order='C', 
                                dtype='uint8', compressor=compressor)
                cells_array.append(image[np.newaxis, :])
        
        elif event_type == 'dot_coordinates':
            dots_filename = position_dir / Path('dots.hdf5')

            # write data into grups that you can index into
            # for example 'timepoint/raw_coords' and 'timepoint/rotated_coords' will give
            # the coordiants on raw image and rotated image
            if 'raw_coords' not in event_data.keys():
                raise KeyError
            if 'rotated_coords' not in event_data.keys():
                raise KeyError
            with h5py.File(dots_filename, 'a') as f:
                f.create_dataset(str(event_data['timepoint']) + '/raw_coords', data=event_data['raw_coords'])
                f.create_dataset(str(event_data['timepoint']) + '/rotated_coords', data=event_data['rotated_coords'])
        
        elif event_type == 'forkplot_data':
            if param.Forkplots.polars is False:
                forks_filename = position_dir / Path('forks.csv')

                if not forks_filename.exists():
                    header=True
                else:
                    header=False
                
                if event_data is not None:
                    fork_stats_table = pd.DataFrame(event_data['fork_data'])
                    fork_stats_table.to_csv(forks_filename, mode='a', index=False, header=header)
                else:
                    raise Exception("Fork plot data write failed")
            elif param.Forkplots.polars is True:
                forks_filename = position_dir / Path('forks.parquet')

                fork_data = event_data['fork_data']
                if len(fork_data) == 0:
                    return
                else:
                    # make dataframe in polars
                    d = {key: [] for key in fork_data[0].keys()}
                    for i in range(len(fork_data)):
                        for key, value in fork_data[i].items():
                            d[key].append(value)
                    df = pl.DataFrame(d)
                    # may be there is a better way.. faster for larger dataset
                    if forks_filename.exists():
                        existing_df = pl.read_parquet(forks_filename)
                        concat_df = pl.concat([existing_df, df])

                        concat_df.write_parquet(forks_filename, partition_by="trap")

                    else:
                        df.write_parquet(forks_filename, partition_by="trap")




    except KeyError:
        sys.stdout.write("Writing failed due to lack of position or time key in data ...\n")
        sys.stdout.flush()
    except Exception as e:
        sys.stdout.write(f"Writing failed due to {e} for data {event_data} ...\n")
        sys.stdout.flush()

def read_files(read_type, param, position, channel_no, max_imgs=20):
    """
    Read data from disk, useful for plotting in the UI.
    """
    try:
        save_dir = Path(param.Save.directory)

        if read_type == 'phase':
            phase_dir = save_dir / Path('Pos' + str(position))  / Path('phase')
            # read all filenames
            phase_filenames = sorted(list(phase_dir.glob('*.tiff')))

            last_key = int(phase_filenames[-1].name.split('_')[-1].split('.')[0])
            
            last_phase_img = imread(phase_filenames[-1])
            height, _ = last_phase_img.shape

            barcode_data = read_from_db('barcode_locations', save_dir, position=position, timepoint=last_key)

            channel_location = barcode_data['trap_locations'][channel_no]

            # find the left and right barcode of the requested channel
            for i, barcode in enumerate(barcode_data['barcode_locations'], 0):
                if ((barcode[0] + barcode[2])/2 > channel_location):
                    break
            left_barcode = barcode_data['barcode_locations'][i-1]
            right_barcode = barcode_data['barcode_locations'][i]
            left_barcode_img = last_phase_img[int(left_barcode[1]): int(left_barcode[3]), int(left_barcode[0]): int(left_barcode[2])]
            right_barcode_img = last_phase_img[int(right_barcode[1]): int(right_barcode[3]), int(right_barcode[0]): int(right_barcode[2])]

            # grab the required number of images
            if max_imgs is not None:
                files_to_iter = phase_filenames[-max_imgs:]
            else:
                files_to_iter = phase_filenames

            trap_width = param.Save.trap_width
            full_img = np.zeros((height, len(files_to_iter) * trap_width))
            for i, filename in enumerate(files_to_iter, 0):
                phase_slice = imread(filename)[:, channel_location-(trap_width//2): channel_location+(trap_width//2)]
                full_img[:, (i) * trap_width : (i+1) * trap_width] = phase_slice

            return {
                'image' : full_img,
                'left_barcode': left_barcode_img,
                'right_barcode': right_barcode_img,
            }
        elif read_type == 'segmented_cells_by_trap':
            
            # read from the .zarr 
            cells_filename = save_dir / Path('Pos' + str(position)) / Path('cells.zarr')

            trap_width = param.Save.trap_width

            data = zarr.convenience.open(cells_filename, mode='r')
            n_slices, height, width = data.shape
            last_key = n_slices - 1

            last_phase_img_filename = save_dir / Path('Pos' + str(position)) / Path('phase') / Path('phase_' + str(last_key).zfill(4) + '.tiff')
            barcode_data = read_from_db('barcode_locations', save_dir, position=position, timepoint=last_key)
            
            channel_location = barcode_data['trap_locations'][channel_no]
            for i, barcode in enumerate(barcode_data['barcode_locations'], 0):
                if (((barcode[0] + barcode[2])/2) > channel_location):
                    break
            left_barcode =  barcode_data['barcode_locations'][i-1]
            right_barcode = barcode_data['barcode_locations'][i]
            last_phase_img = imread(last_phase_img_filename)

            left_barcode_img = last_phase_img[int(left_barcode[1]): int(left_barcode[3]), int(left_barcode[0]): int(left_barcode[2])]
            right_barcode_img = last_phase_img[int(right_barcode[1]): int(right_barcode[3]), int(right_barcode[0]): int(right_barcode[2])]

            if max_imgs is not None:
                full_img = data[:, :, (channel_no) * trap_width : (channel_no+1) * trap_width]
                full_img = np.hstack(full_img)
            else:
                full_img = data[-max_imgs:, :, channel_no * trap_width: (channel_no+1) * trap_width]
                full_img = np.hstack(full_img)

            return {
                'image': full_img,
                'left_barcode': left_barcode_img,
                'right_barcode': right_barcode_img
            }
            
        elif read_type == 'dots_on_mask':
            # just read the hdf5 file and give dots for the whole trap
            #dots_path  = save_dir / Path('Pos' + str(position)) / Path('dots.hdf5')
            
            # read all the dots and index by trap

                        
            # read from the .zarr 
            cells_filename = save_dir / Path('Pos' + str(position)) / Path('cells.zarr')

            trap_width = param.Save.trap_width

            data = zarr.convenience.open(cells_filename, mode='r')
            n_slices, height, width = data.shape
            last_key = n_slices - 1

            last_phase_img_filename = save_dir / Path('Pos' + str(position)) / Path('phase') / Path('phase_' + str(last_key).zfill(4) + '.tiff')
            barcode_data = read_from_db('barcode_locations', save_dir, position=position, timepoint=last_key)
            
            channel_location = barcode_data['trap_locations'][channel_no]
            for i, barcode in enumerate(barcode_data['barcode_locations'], 0):
                if (((barcode[0] + barcode[2])/2) > channel_location):
                    break
            left_barcode =  barcode_data['barcode_locations'][i-1]
            right_barcode = barcode_data['barcode_locations'][i]
            last_phase_img = imread(last_phase_img_filename)

            left_barcode_img = last_phase_img[int(left_barcode[1]): int(left_barcode[3]), int(left_barcode[0]): int(left_barcode[2])]
            right_barcode_img = last_phase_img[int(right_barcode[1]): int(right_barcode[3]), int(right_barcode[0]): int(right_barcode[2])]

            if max_imgs is not None:
                full_img = data[:, :, (channel_no) * trap_width : (channel_no+1) * trap_width]
                full_img = np.hstack(full_img)
                indices_to_iter = range(data.shape[0])
            else:
                full_img = data[-max_imgs:, :, channel_no * trap_width: (channel_no+1) * trap_width]
                full_img = np.hstack(full_img)
                indices_to_iter = range(data.shape[0]-max_imgs, data.shape[0])
            

            dots_filename = save_dir / Path('Pos' + str(position)) / Path('dots.hdf5')
            # figure out dots on this image
            dot_keys = []
            timepoints = []
            channel_locations = [] # trap locations change for each image so you really want to grab them
            for index in indices_to_iter:
                timepoint = index
                timepoints.append(timepoint)
                dot_keys.append(str(timepoint) + '/raw_coords')

                barcode_data = read_from_db('barcode_locations', save_dir, position=position, timepoint=timepoint)
                channel_location = barcode_data['trap_locations'][channel_no]
                channel_locations.append(channel_location)

            # loop over and filter dots that fall inside the trap
            dot_data = []
            with h5py.File(dots_filename, 'r') as f:
                for i, key in enumerate(dot_keys, 0):
                    single_array = f[key][:]
                    indices = np.where(np.logical_and(single_array[:, 1] > channel_locations[i] - (trap_width//2),
                                         single_array[:, 1] < channel_locations[i] + (trap_width//2)))[0]
                    single_array_trap_filtered = single_array[indices]
                    single_array_trap_filtered[:, 1] = single_array_trap_filtered[:, 1] - channel_locations[i] + trap_width//2 + (i * trap_width)
                    dot_data.append(single_array_trap_filtered)

            dots = np.vstack(dot_data) 


            return {
                'image': full_img,
                'left_barcode': left_barcode_img,
                'right_barcode': right_barcode_img,
                'dots': dots
            }
            
        elif read_type == 'fluor':
            fluor_dir = save_dir / Path('Pos' + str(position)) / Path('fluor')

            # use phase image to update the barcodes
            phase_dir = save_dir / Path('Pos' + str(position)) / Path('phase')
            # read all files 
            fluor_filenames = sorted(list(fluor_dir.glob('*.tiff')))

            last_key = int(fluor_filenames[-1].name.split('_')[-1].split('.')[0])

            last_fluor_img = imread(fluor_filenames[-1])
            height, _ = last_fluor_img.shape

            last_phase_img_filename = phase_dir / Path('phase_' + str(last_key).zfill(4) + '.tiff')
            last_phase_img = imread(last_phase_img_filename)

            barcode_data = read_from_db('barcode_locations', save_dir, position=position, timepoint=last_key)
            channel_location = barcode_data['trap_locations'][channel_no]

            for i, barcode in enumerate(barcode_data['barcode_locations'], 0):
                if (((barcode[0] + barcode[2])/2) > channel_location):
                    break
            left_barcode =  barcode_data['barcode_locations'][i-1]
            right_barcode = barcode_data['barcode_locations'][i]
            left_barcode_img = last_phase_img[int(left_barcode[1]): int(left_barcode[3]), int(left_barcode[0]): int(left_barcode[2])]
            right_barcode_img = last_phase_img[int(right_barcode[1]): int(right_barcode[3]), int(right_barcode[0]): int(right_barcode[2])]

            if max_imgs is not None:
                files_to_iter = fluor_filenames[-max_imgs:]
            else:
                files_to_iter = fluor_filenames


            trap_width = param.Save.trap_width
            full_img = np.zeros((height, len(files_to_iter) * trap_width))
            for i, filename in enumerate(files_to_iter, 0):
                fluor_slice = imread(filename)[:, channel_location - (trap_width//2) : channel_location + (trap_width//2)]
                full_img[:, (i) * trap_width : (i+1) * trap_width] = fluor_slice

            dots_filename = save_dir / Path('Pos' + str(position)) / Path('dots.hdf5')
            # figure out dots on this image
            dot_keys = []
            timepoints = []
            channel_locations = [] # trap locations change for each image so you really want to grab them
            for filename in files_to_iter:
                timepoint = int(filename.stem.split('_')[-1])
                timepoints.append(timepoint)
                dot_keys.append(str(timepoint) + '/raw_coords')

                barcode_data = read_from_db('barcode_locations', save_dir, position=position, timepoint=timepoint)
                channel_location = barcode_data['trap_locations'][channel_no]
                channel_locations.append(channel_location)

            # loop over and filter dots that fall inside the trap
            dot_data = []
            with h5py.File(dots_filename, 'r') as f:
                for i, key in enumerate(dot_keys, 0):
                    single_array = f[key][:]
                    indices = np.where(np.logical_and(single_array[:, 1] > channel_locations[i] - (trap_width//2),
                                         single_array[:, 1] < channel_locations[i] + (trap_width//2)))[0]
                    single_array_trap_filtered = single_array[indices]
                    single_array_trap_filtered[:, 1] = single_array_trap_filtered[:, 1] - channel_locations[i] + trap_width//2 + (i * trap_width)
                    dot_data.append(single_array_trap_filtered)

            dots = np.vstack(dot_data) 

            return {
                'image': full_img,
                'left_barcode': left_barcode_img,
                'right_barcode': right_barcode_img,
                'dots': dots
            }

        elif read_type == 'all_forks':

            #if os.name == 'nt':
             #   use_pyarrow = True
            #else:
             #   use_pyarrow = False

            use_pyarrow = False
            if param.Forkplots.polars is False:
                forks_filenames = [position_dir / Path('forks.csv') for position_dir in list(save_dir.glob('Pos*'))]
                dataframes = [pd.read_csv(filename) for filename in forks_filenames]
                data = pd.concat(dataframes, ignore_index=True)

            else:
                # read parquet files
                fork_filenames = glob.glob(str(save_dir / Path('Pos[0-9]*/forks.parquet')))
                columns_to_extract = ['area', 'length', 'normalized_internal_x', 'normalization_counts']
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

            heatmap, mean_cell_lengths, abins, lbins, extent = fast_generate_fork_plot(areas, lengths, longs, counts,
                            bin_scale=bin_scale,
                            pixel_size=pixel_size,
                            heatmap_threshold=heatmap_threshold)

            init_area = get_bulk_init_area(areas, counts, longs, lengths, min_length=min_length, 
                                           pixel_size=pixel_size, arb_div_area=arb_div_area)
            
            area_bins_around_init, lbins_around_init, heatmap_around_init, mean_cell_lengths_around_init, abins_inds_around_init, lbins_inds_around_init = slice_fork_plot_around_init(abins, lbins, heatmap, mean_cell_lengths, init_area, init_area_cv)
            
            #Flat heatmap
            flat_heatmap_init = heatmap_around_init.flatten()

            #Calculation for Moran's I
            moran_weight = pysal.lib.weights.lat2W(heatmap_around_init.shape[0], heatmap_around_init.shape[-1], rook=True)

            #Calculation for energy test
            e_dists = energy_dists(lbins_around_init, area_bins_around_init)

            return {
                #Maybe do not need to pass full fork plot parameters, 
                #but will do so for now so we do not need to rewrite 
                #code if we end up using those instead of the sliced 
                #fork plots
                'heatmap': heatmap,
                'mean_cell_lengths': mean_cell_lengths,
                'extent': extent,
                'init_area': init_area,
                'area_bins_around_init': area_bins_around_init,
                'lbins_around_init': lbins_around_init, 
                'heatmap_around_init': heatmap_around_init,
                'mean_cell_lengths_around_init': mean_cell_lengths_around_init,
                'abins_inds_around_init': abins_inds_around_init,
                'lbins_inds_around_init': lbins_inds_around_init,
                'flat_heatmap_init': flat_heatmap_init,
                'moran_weight': moran_weight,
                'e_dists': e_dists
            }

        elif read_type == 'single_trap_data_forks':

            bin_scale = param.Forkplots.bin_scale
            heatmap_threshold = param.Forkplots.heatmap_threshold
            pixel_size = param.Forkplots.pixel_size

            #if os.name == 'nt':
             #   use_pyarrow = True
            #else:
             #   use_pyarrow = False
            use_pyarrow = False
            if param.Forkplots.polars is False:
                #Add so that you do not need to regenerate the bins after having done the full one
                all_forks_filenames = [position_dir / Path('forks.csv') for position_dir in list(save_dir.glob('Pos*'))]
                dataframes = [pd.read_csv(filename) for filename in all_forks_filenames]
                all_data = pd.concat(dataframes, ignore_index=True)

            else:
                # read parquet files
                fork_filenames = glob.glob(str(save_dir / Path('Pos[0-9]*/forks.parquet')))
                columns_to_extract = ['area', 'length', 'normalized_internal_x', 'normalization_counts']
                all_data = pl.read_parquet(fork_filenames, use_pyarrow=use_pyarrow, columns=columns_to_extract)


            abins, lbins = generate_abins_lbins(all_data['area'].to_numpy(),
                                    all_data['length'].to_numpy(),
                                    all_data['normalized_internal_x'].to_numpy(),
                                    pixel_size=pixel_size,
                                    bin_scale=bin_scale)

            if param.Forkplots.polars is False:
                forks_filename = save_dir / Path('Pos' + str(position)) / Path('forks.csv')
                data = pd.read_csv(forks_filename)


                # filter for trap no
                trap_data = data[data['trap'] == channel_no]
            else:
                # set filename 
                # read parquet files
                forks_filename =  save_dir / Path('Pos' + str(position)) / Path('forks.parquet/trap=' + str(channel_no))
                columns_to_extract = ['area', 'length', 'normalized_internal_x', 'normalization_counts']
                trap_data = pl.read_parquet(forks_filename, use_pyarrow=use_pyarrow, columns=columns_to_extract)


                # read data

            areas = trap_data['area'].to_numpy()
            lengths = trap_data['length'].to_numpy()
            longs = trap_data['normalized_internal_x'].to_numpy()
            counts = trap_data['normalization_counts'].to_numpy()
            heatmap_trap, mean_cell_lengths_trap, _, _, _ = generate_fork_plot(areas, lengths, longs, counts, abins=abins, lbins=lbins,
                            bin_scale=bin_scale,
                            pixel_size=pixel_size,
                            heatmap_threshold=heatmap_threshold)

            return {
                'heatmap_trap': heatmap_trap,
                'mean_cell_lengths_trap': mean_cell_lengths_trap,
                #'extent': extent
            }


    except Exception as e:
        sys.stdout.write(f"Reading data failed due to {e} for {read_type} ...\n")
        sys.stdout.flush()