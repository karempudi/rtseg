
import numpy as np
import sys
from pathlib import Path
from skimage.io import imsave, imread
import zarr
from numcodecs import Zlib
from rtseg.utils.db_ops import read_from_db
from tifffile import imwrite # used for finer compression things
import h5py
import pandas as pd

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
            if max_imgs is None:
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

            if max_imgs is None:
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

            if max_imgs is None:
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
            
            return None


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

            if max_imgs is None:
                files_to_iter = fluor_filenames[-max_imgs:]
            else:
                files_to_iter = fluor_filenames

            trap_width = param.Save.trap_width
            full_img = np.zeros((height, len(files_to_iter) * trap_width))
            for i, filename in enumerate(files_to_iter, 0):
                fluor_slice = imread(filename)[:, channel_location - (trap_width//2) : channel_location + (trap_width//2)]
                full_img[:, (i) * trap_width : (i+1) * trap_width] = fluor_slice

            return {
                'image': full_img,
                'left_barcode': left_barcode_img,
                'right_barcode': right_barcode_img
            }

            

    except Exception as e:
        sys.stdout.write(f"Reading data failed due to {e} for {read_type} ...\n")
        sys.stdout.flush()