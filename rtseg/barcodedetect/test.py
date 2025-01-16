import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from art import tprint
from datetime import datetime
from rtseg.utils.param_io import load_params, save_params
from rtseg.barcodedetect.datasets import BarcodeTestDataset
from rtseg.barcodedetect.transforms import YoloTestAugmentations
from rtseg.barcodedetect.networks import YOLOv3
from rtseg.barcodedetect.utils import non_max_suppression, plot_results_batch, to_cpu
from rtseg.barcodedetect.utils import outputs_to_bboxes
import torch
from torch.utils.data import DataLoader
from rtseg.utils.hardware import get_device_str
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Region detection testing arguments')
    
    parser.add_argument('-p', '--param_file', 
                        help='Specify the param file for testing region detection network',
                        required=True)

    parser.add_argument('-d', '--device', default=None, 
                        help='Specify the device string (cpu, cuda, cuda:0, or cuda:1)',
                        type=str, required=False)
    
    parser.add_argument('-w', '--num_workers_override', default=6,
                        help='Override number of workers for pytorch dataloader.',
                        type=int, required=False)
    
    parser.add_argument('-i', '--images_dir_override', default=None,
                        help='Set directory containing images to override',
                        type=str, required=False)
    
    parser.add_argument('-s', '--save_dir_override', default=None,
                        help='Set directory to save plots to override',
                        type=str, required=False)
    
    parser.add_argument('-m', '--model_path', default=None,
                        help='Set model path to override',
                        type=str, required=False)
    
    args = parser.parse_args()

    return args

def test_model(param_file: str, device_overwrite: str = None,
                num_workers_override: int = None,
                images_dir_override: str = None,
                save_dir_override: str = None,
                model_path_override: str = None):
    """
    Test a region detection (barcode) model based on the parameters given

    Args:
        param_file (str): parameter file path, includes everything needed to 
            load a yolo series model for bounding box detection on the images
        device_overwrite (str): overwrite cuda device specified in the param_file
        num_workers_override (int): overwrite num of workers in pytorch dataloaders
        images_dir_override (str): overwrite images_dir in the param file 
        save_dir_override (str) : overwrite save_dir in the param file 

        Note:
            the parameters file is the same as yolo training config, but 
            all the parameters will be read from the testing section of the 
            file instead, everything else will be ignored

    """
    # Loading the param
    param_file = Path(param_file)
    if not param_file.exists():
        raise FileNotFoundError(f"Parameter file {str(param_file)} doesn't exist")

    # make a param namespace and fill in the defaults
    param = load_params(param_file, ref_type='barcode')

    # set hardware device to test one
    if device_overwrite is not None:
        param.Testing.Hardware.device = device_overwrite
        device = device_overwrite
    else:
        device = param.Testing.Hardware.device

    # check for cuda
    if torch.cuda.is_available():
        _, device_idx = get_device_str(device)
        if device_idx is not None:
            torch.cuda.set_device(device)
    else:
        device = 'cpu'
    
    torch.set_num_threads(param.Testing.Hardware.torch_threads)

    if num_workers_override is not None:
        param.Testing.Hardware.num_workers = num_workers_override
    
    if images_dir_override is not None:
        param.Testing.images_dir = images_dir_override

    if save_dir_override is not None:
        param.Testing.save_dir = save_dir_override

    if model_path_override is not None:
        param.Testing.model_path = model_path_override

    expt_id = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    expt_dir = Path(param.Testing.save_dir) / Path(expt_id)

    if not expt_dir.parent.exists():
        expt_dir.parent.mkdir()
    
    if not expt_dir.exists():
        expt_dir.mkdir()

    param_used_save = expt_dir / Path('testing_params').with_suffix(param_file.suffix)
    save_params(param_used_save, param.Testing)


    # setup dataset and dataloader
    if param.Testing.transformations.type == 'YoloTestAugmentations':
        dataset_transformations = YoloTestAugmentations()

    print(param.Testing)

    test_ds = BarcodeTestDataset(images_dir=param.Testing.images_dir, transform=dataset_transformations)
    test_dl = DataLoader(
        dataset = test_ds,
        batch_size=param.Testing.batch_size,
        drop_last=False,
        shuffle=False,
        num_workers=param.Testing.Hardware.num_workers,
        pin_memory=False,
        collate_fn=test_ds.collate_fn
    )
    model, anchors, strides = setup_tester(param, device)

    # Test loop
    model.eval()
    for batch_i, (paths, images) in enumerate(tqdm(test_dl, desc="Test dataset: ")):
        #batches_test_done = len(test_dl) * epoch + batch_i
        with torch.no_grad():
            predictions = model(images.to(device, non_blocking=True))
            bboxes = outputs_to_bboxes(predictions, anchors, strides)
            bboxes_cleaned = non_max_suppression(bboxes, conf_thres=param.Testing.thresholds.conf,
                                        iou_thres=param.Testing.thresholds.iou)
            # each image in the batch has a bbox array in the list
            bboxes_numpy = [bbox.numpy() for bbox in bboxes_cleaned]
            # write figures to a directory to save
            batch_fig_handles = plot_results_batch(to_cpu(images).numpy(), bboxes_numpy)
            for i, figure in enumerate(batch_fig_handles, 0):
                save_path = Path(expt_dir) / Path(Path(paths[i]).stem + '.png')
                figure.savefig(save_path, bbox_inches='tight')
                plt.close(figure)


def setup_tester(param, device):
    """
    Sets up all the things needed for testing a network and returns them

    Args:
        param (RecursiveNamespace) : contains all parameters needed
        device: device to send the model to already
    """
    models_available = {
        'YOLOv3': YOLOv3
    }

    model = models_available[param.Testing.architecture]
    anchor_sizes = param.Testing.model_params.anchors.sizes
    strides = param.Testing.model_params.anchors.strides
    num_classes = param.Testing.model_params.num_classes

    anchors_list = [[anchor_sizes[i], anchor_sizes[i+1], anchor_sizes[i+2]] for i in range(0, len(anchor_sizes), 3)]

    model = model.parse(anchors=anchors_list, num_classes=num_classes).to(device=device)

    # Load model params from file
    model.load_state_dict(torch.load(param.Testing.model_path))

    anchors_t = tuple(torch.tensor(anch).float().to(device=device) for anch in anchors_list)
    strides_t = tuple(torch.tensor(stride).to(device=device) for stride in strides)

    return model, anchors_t, strides_t


def main():
    print("Hello from region detection testing")
    tprint("BARCODE TEST")
    args = parse_args()

    test_model(args.param_file,
                device_overwrite=args.device,
                num_workers_override=args.num_workers_override,
                images_dir_override=args.images_dir_override,
                save_dir_override=args.save_dir_override,
                model_path_override=args.model_path)

if __name__ == "__main__":
    main()
