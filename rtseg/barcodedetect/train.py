import argparse
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from rtseg.utils.param_io import load_params, save_params
from rtseg.barcodedetect.logger import SummaryWriter
from rtseg.barcodedetect.datasets import BarcodeDataset, BarcodeTestDataset
from rtseg.barcodedetect.transforms import YoloAugmentations, YoloTestAugmentations
from rtseg.barcodedetect.networks import YOLOv3
from rtseg.barcodedetect.loss import compute_yolo_loss
from rtseg.barcodedetect.utils import CosineWarmupScheduler, outputs_to_bboxes
from rtseg.barcodedetect.utils import non_max_suppression, plot_results_batch, to_cpu
import torch
from torch.utils.data import DataLoader
from rtseg.utils.hardware import get_device_str
from art import tprint
from torch.utils.data import random_split
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Barcode detection Training Arguments")

    parser.add_argument('-p', '--param_file',
                        help='Specify the param file for training barcode detection network',
                        required=True)
    parser.add_argument('-d', '--device', default=None,
                        help='Specify the device string (cpu, cuda, cuda:0, or cuda:1)',
                        type=str, required=False)

    parser.add_argument('-w', '--num_workers_override', default=None,
                        help='Override number of workers for pytorch dataloader.',
                        type=int, required=False)
    
    parser.add_argument('-l', '--log_dir', default='barcode_logs',
                        help='Set directory name where you want the logs to go in',
                        required=False, type=str)
    
    parser.add_argument('-c', '--log_comment', default=None,
                        help='Added a log comment to the run',
                        type=str, required=False)
    
    args = parser.parse_args()
    
    return args

    
def train_model(param_file: str, device_overwrite: str = None,
                num_workers_overwrite: int = None,
                log_dir: str = 'barcode_logs', log_comment: str = None
                ):
    
    """
    Train are region detection model based on the parameters given
    
    Args:
        param_file (str): parameter file path, includes everything need to
            train a yolo series models for drawing bounding boxes around
            the regions containing barcdes
        device_overwrite (str): overwrite cuda device specified in the param_file 
        num_workers_overwrite (int): overwrite num of workers in pytorch dataloaders
        log_dir (str): directory for logs used by tensorboard
        log_comment (str): comment to the experiment

    """

    # Loading parameters for network training from the parameter file
    param_file = Path(param_file)
    if not param_file.exists():
        raise FileNotFoundError(f"Parameter file {str(param_file)} doesn't exist")
    
    # make a param namespace and fill in defaults
    param = load_params(param_file, ref_type='barcodedetect')
    print(param)

    # Load ckpt if the training stops and you need to restart
    if param.Checkpoints.load_ckpt is False:
        expt_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        from_ckpt = False
    else:
        from_ckpt = True
        expt_id = Path(param.Checkpoints.ckpt_filename).parent.name

    if log_comment:
        expt_id = expt_id + '-' + log_comment
    
    if not from_ckpt:
        # create a new expt dir
        expt_dir = Path(param.Save.directory) / Path(expt_id)
    else:
        expt_dir = Path(param.Checkpoints.ckpt_filename).parent
    
    if not expt_dir.parent.exists():
        expt_dir.parent.mkdir()

    
    # create dir for expt if not loaded from ckpt
    if not from_ckpt:
        expt_dir.mkdir(exist_ok=False)

    model_out = expt_dir / Path('model.pth') 
    ckpt_path = expt_dir / Path('ckpt.pth')

    # save parameter file used in the directory of the expt
    param_in_save = expt_dir / Path('training_run_set').with_suffix(param_file.suffix)
    shutil.copy(param_file, param_in_save)

    # set hardware device to train
    if device_overwrite is not None:
        param.Hardware.device = device_overwrite
        device = device_overwrite
    else:
        device = param.Hardware.device
    
    # # check for cuda
    if torch.cuda.is_available():
        _, device_idx = get_device_str(device)
        if device_idx is not None:
            torch.cuda.set_device(device) 
    else:
        device = 'cpu'

    if num_workers_overwrite is not None:
        param.Hardware.num_workers = num_workers_overwrite
    
    torch.set_num_threads(param.Hardware.torch_threads)

    # training run params after filling the defualts
    param_used_save = expt_dir / Path('training_run_used').with_suffix(param_file.suffix)
    save_params(param_used_save, param)


    # Log system
    # Use tensorboard on this directory to view logs of all the experimental runs
    log_dir_path = Path(param.Save.directory).parent / log_dir
    if not log_dir_path.exists():
        log_dir_path.mkdir(exist_ok=False)

    # setup tensorboard logger, what to write to tensorboard add things to logger as you train
    logger = SummaryWriter(log_dir=log_dir_path/expt_id)
    
    # train and validation datasets
    if param.Datasets.transformations.type == 'YoloAugmentations':
        dataset_transformations = YoloAugmentations()
    dataset = BarcodeDataset(data_dir=param.Datasets.directory, transform=dataset_transformations) 
    len_dataset = len(dataset)
    train_len = int(len_dataset * param.Datasets.train.percentage)
    val_len = len_dataset - train_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len]) 

    # Test dataset on which the predictions are plotted
    #if param.Datasets.test.transformations == 'YoloTestAugmentations':
    #    test_dataset_transformations = YoloTestAugmentations()
    
    test_ds = BarcodeTestDataset(images_dir= param.Datasets.test.directory, transform=YoloTestAugmentations())
    print(f"Length of train dataset: {len(train_ds)} , validation: {len(val_ds)}, test: {len(test_ds)}")

    # setup dataloaders
    train_dl, val_dl, test_dl = setup_dataloader(param, train_ds, val_ds, test_ds)

    #train_batch = next(iter(train_dl))
    #val_batch = next(iter(val_dl))
    #test_batch = next(iter(test_dl))
    #print(train_batch[0], train_batch[1].shape, train_batch[2].shape)
    #print(val_batch[0], val_batch[1].shape, val_batch[2].shape)
    #print(test_batch[0], test_batch[1].shape, test_batch[2].shape)

    # setup trainer

    model, optimizer, criterion, lr_scheduler, anchors, strides = setup_trainer(param, 
                    logger, model_out, ckpt_path, device) 

    # train loop
    nEpochs = param.HyperParameters.epochs
    print(f"Model training for epochs: {nEpochs}")
    print("\n--- Training barcode detection model ---")
    current_lr = lr_scheduler.get_last_lr()[0]
    logger.add_scalar(param.HyperParameters.optimizer.name + '/lr', current_lr, 0)


    best_val_loss = 1_000_000.
    for epoch in range(1, nEpochs + 1):
        model.train()
        epoch_loss = []
        for batch_i, (_, images, targets) in enumerate(tqdm(train_dl, desc=f"Training Epcoh {epoch}")):
            optimizer.zero_grad()
            batches_done = len(train_dl) * epoch + batch_i
            images = images.to(device, non_blocking=True)
            targets = targets.to(device)

            predictions = model(images)
            loss, loss_parts = criterion(predictions, targets, anchor_boxes=anchors, strides=strides)
            loss.backward()
            epoch_loss.append(loss.item())
            optimizer.step()
            loss_dict = {
                'iou_loss': float(loss_parts[0]),
                'obj_loss': float(loss_parts[1]),
                'class_loss': float(loss_parts[2]),
                'loss': loss.cpu().item()
            }
            logger.add_scalar_dict('train/', loss_dict, global_step=batches_done)

        lr_scheduler.step() 
        current_lr = lr_scheduler.get_last_lr()[0]
        logger.add_scalar(param.HyperParameters.optimizer.name + '/lr', current_lr, batches_done)

        print(f"Epoch: {epoch} Train loss: {np.mean(epoch_loss): 0.4f} lr: {lr_scheduler.get_last_lr()[0] :.6f}")

        # validation loop
        model.eval()
        epoch_val_loss = []
        for batch_i, (_, images, targets) in enumerate(tqdm(val_dl, desc=f"Validation Epoch {epoch}")):
            batches_val_done = len(val_dl) * epoch + batch_i
            images = images.to(device, non_blocking=True)
            targets = targets.to(device)
            predictions = model(images)
            loss_val, loss_val_parts = criterion(predictions, targets, anchor_boxes=anchors, strides=strides)
            epoch_val_loss.append(loss_val.item())
            loss_dict_val = {
                'iou_loss': float(loss_val_parts[0]),
                'obj_loss': float(loss_val_parts[1]),
                'class_loss': float(loss_val_parts[2]),
                'loss': loss_val.cpu().item()
            }
            logger.add_scalar_dict('validation/', loss_dict_val, global_step=batches_val_done)
        mean_epoch_val_loss = np.mean(epoch_val_loss)
        print(f"Epoch: {epoch} Validation loss: {mean_epoch_val_loss: 0.4f}")

        # save model if 
        if mean_epoch_val_loss < best_val_loss:
            best_val_loss = mean_epoch_val_loss
            torch.save(model.state_dict(), model_out)

        # Test loop
        model.eval()
        for batch_i, (paths, images) in enumerate(tqdm(test_dl, desc=f"Test Epoch {epoch}")):
            batches_test_done = len(test_dl) * epoch + batch_i
            with torch.no_grad():
                predictions = model(images.to(device, non_blocking=True))
                bboxes = outputs_to_bboxes(predictions, anchors, strides)
                bboxes_cleaned = non_max_suppression(bboxes, conf_thres=param.Datasets.test.conf_thres,
                                            iou_thres=param.Datasets.test.iou_thres)
                # each image in the batch has a bbox array in the list
                bboxes_numpy = [bbox.numpy() for bbox in bboxes_cleaned]
                if batch_i == len(test_dl) - 1:
                    batch_fig_handles = plot_results_batch(to_cpu(images).numpy(), bboxes_numpy)
                    for i, figure in enumerate(batch_fig_handles, 0):
                        logger.add_figure('test/fig' + str(i), figure, global_step=batches_test_done)
                # write figures to a directory to save
                elif epoch == nEpochs:
                    batch_fig_handles = plot_results_batch(to_cpu(images).numpy(), bboxes_numpy)
                    for i, figure in enumerate(batch_fig_handles, 0):
                        save_path = Path(param.Datasets.test.save_directory) / Path(Path(paths[i]).stem + '.png')
                        figure.savefig(save_path, bbox_inches='tight')
                        plt.close(figure)

        # Do more metrics on test and/or validation data


    print("\n---- Training done ----\n")
    
    # Metrics 

    # saving the model

    

    # test loop



def setup_trainer(param, logger, model_out, ckpt_path, device):
    """
    Sets up all the things needed for training of a network and returns them

    Args:
        param (RecursiveNamespace): contains all the parameters needed
        logger (torch.utils.tensorboard.SummaryWriter): a custom tensorboard logger
        model_out (pathlib.Path): model save path
        ckpt_path (pathlib.Path): ckpt save path
        device (torch.device): device onto which the network is loaded

    Returns:
        model: an instance of the model loaded on the appropriate device
        optimizer: an instance of optimizer based on Hyperparameters
        criterion: a loss function used to calculate loss
        lr_scheduler: learning rate scheduler based on hyperparameters provided
        anchors : anchors in tensor form used by the loss function
        strides: strides in tensor form used by the loss function

    """
    models_available = {
        'YOLOv3': YOLOv3
    }

    model = models_available[param.HyperParameters.architecture]
    anchor_sizes = param.HyperParameters.model_params.anchors.sizes
    strides = param.HyperParameters.model_params.anchors.strides
    num_classes = param.HyperParameters.model_params.num_classes
    #print(f"Anchors are .... {anchor_sizes} .. strides: {strides} .. num_classes: {num_classes}")

    anchors_list = [[anchor_sizes[i], anchor_sizes[i+1], anchor_sizes[i+2]] for i in range(0, len(anchor_sizes), 3)]

    model = model.parse(anchors=anchors_list, num_classes=num_classes).to(device=device)

    anchors_t = tuple(torch.tensor(anch).float().to(device=device) for anch in anchors_list)
    strides_t = tuple(torch.tensor(stride).to(device=device) for stride in strides)
    #print(f"Anchor tensors: {anchors_t}")
    #print(f"Stride_tensors: {strides_t}")

    if param.HyperParameters.loss == 'yolo_loss':
        criterion = compute_yolo_loss

    if param.HyperParameters.optimizer.name == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(),
                     lr=param.HyperParameters.optimizer.learning_rate,
                     weight_decay=param.HyperParameters.optimizer.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(),
                    lr=param.HyperParameters.optimizer.learning_rate,
                    weight_decay=param.HyperParameters.optimizer.weight_decay,
                    momentum=param.HyperParameters.optimizer.momentum)

    if param.HyperParameters.scheduler.name == "CosineWarmup":
        lr_scheduler = CosineWarmupScheduler(optimizer, 
                    warmup=param.HyperParameters.scheduler.warmup,
                    max_iters=param.HyperParameters.scheduler.max_iters)
    else:
        lr_scheduler = None

    print(f"Optimizer {param.HyperParameters.optimizer.name} lr: {param.HyperParameters.optimizer.learning_rate}")
    print(f"Scheduler: {param.HyperParameters.scheduler.name}, warmup: {param.HyperParameters.scheduler.warmup}")

    return model, optimizer, criterion, lr_scheduler, anchors_t, strides_t

def setup_dataloader(param, train_ds, val_ds=None, test_ds=None):
    """
    Setup dataloaders for the datasets
    Args:
        param (RecursiveNamespace): 
        train_ds (torch.utils.Dataset): an instance of Dataset class
        val_ds (optional): an instance of Dataset class
        test_ds (optional): an instance of Dataset class
    """

    train_dl = DataLoader(
        dataset=train_ds,
        batch_size=param.HyperParameters.train_batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=param.Hardware.num_workers,
        pin_memory=True,
        collate_fn=train_ds.dataset.collate_fn
    )

    if val_ds is not None:
        val_dl = DataLoader(
            dataset=val_ds,
            batch_size=param.HyperParameters.validation_batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=param.Hardware.num_workers,
            pin_memory=False,
            collate_fn=val_ds.dataset.collate_fn
        )
    else:
        val_dl = None    

    if test_ds is not None:
        test_dl = DataLoader(
            dataset=test_ds,
            batch_size=param.HyperParameters.test_batch_size,
            drop_last=True,
            shuffle=False,
            num_workers=param.Hardware.num_workers,
            pin_memory=False,
            collate_fn=test_ds.collate_fn
        )
    else:
        test_dl = None
    

    return train_dl, val_dl, test_dl


def main():
    print("Hello from barcode detection training")
    tprint("BARCODE")
    args = parse_args()

    train_model(args.param_file, 
                device_overwrite=args.device,
                num_workers_overwrite=args.num_workers_override,
                log_dir=args.log_dir,
                log_comment=args.log_comment,
                )

if __name__ == "__main__":
    main()