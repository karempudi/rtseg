import argparse
import shutil
from pathlib import Path
from datetime import datetime
from art import tprint
import numpy as np
import matplotlib.pyplot as plt
from rtseg.utils.param_io import load_params, save_params
from rtseg.utils.hardware import get_device_str
from rtseg.oldseg.datasets import MMDatasetUnetDual, MMDatasetUnetTest
from rtseg.oldseg.loss import UnetDualLoss
from rtseg.oldseg.transforms import UnetTestTransforms, UnetTrainTransforms
from rtseg.oldseg.logger import SummaryWriter
import torch
from torch.utils.data import DataLoader, random_split
from rtseg.oldseg.networks import model_dict 
from rtseg.oldseg.utils import CosineWarmupScheduler, to_cpu, plot_results_batch
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Segmentation networks Training Arguments")

    parser.add_argument('-p', '--param_file', 
            help='Specify the param file for training segmentation network.',
            required=True)
    
    parser.add_argument('-d', '--device',
            help='Specify the device string (cpu, cuda, cuda:0 or cuda:1) and overwrite parameters.',
            type=str, required=False)
    
    parser.add_argument('-w', '--num_workers_override', default=None,
            help='Override number of workers of pytorch dataloaders.',
            type=int, required=False)
    
    parser.add_argument('-l', '--log_dir', default='seg_logs',
            help='Set directory name where you want the logs to go in.',
            type=str, required=False)
    
    parser.add_argument('-c', '--log_comment', default=None,
            help='Added a log comment to the run',
            type=str, required=False)

    args = parser.parse_args()

    return args

def train_model(param_file: str, device_overwrite: str = None,
            num_workers_override: int = None,
            log_dir: str = 'seg_logs', log_comment: str = None):
    """
    Train a segmentation model based on the parameters given

    Args:
        param_file (str): parameter file path, includes everything need to 
                train a segmentation network, U-net, Omnipose and their 
                variants for both cell and channel detection
        device_overwrite (str): overwrite cuda device specified in param_file
        num_workers_override (int): overwrite num of workers in pytorch dataloaders
        log_dir (str): directory for logs used by tensorboard
        log_comment (str): comment to the experiment

    """

    # Loading parameters for network training from the parameter file
    param_file = Path(param_file)
    if not param_file.exists():
        raise FileNotFoundError(f"Parameter file {str(param_file)} doesn't exist")
    # make a param namespace and fill in defaults
    param = load_params(param_file, 'oldseg')

    # Load ckpt if the training stops and you need to restart 
    if param.Checkpoints.load_ckpt is False:
        expt_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        from_ckpt = False
    else:
        from_ckpt = True
        expt_id = Path(param.Checkpoints.ckpt_filename).parent.name

    if log_comment:
        expt_id = expt_id + '_' + log_comment

    if not from_ckpt:
        # create new expt dir
        expt_dir = Path(param.Save.directory) / Path(expt_id)
    else:
        expt_dir = Path(param.Checkpoints.ckpt_filename).parent
    
    if not expt_dir.parent.exists():
        expt_dir.parent.mkdir()

    # create dir for expt if not loaded from ckpt
    if not from_ckpt:
        expt_dir.mkdir(exist_ok=False)

    model_out = expt_dir / Path('model.pth')
    # for now we dont' save checkpoints, but we will do it later to restart 
    # training sessions
    #ckpt_path = expt_dir / Path('ckpt.pth')

    # save parameter file used in the directory of the experiment
    param_in_save = expt_dir / Path('training_run_set').with_suffix(param_file.suffix)
    shutil.copy(param_file, param_in_save)
   # set hardware device to train
    if device_overwrite is not None:
        param.Hardware.device = device_overwrite
        device = device_overwrite
    else:
        device = param.Hardware.device
    
    # check for cuda
    if torch.cuda.is_available():
        _, device_idx = get_device_str(device)
        if device_idx is not None:
            torch.cuda.set_device(device) 
    else:
        device = 'cpu'
    
    if num_workers_override is not None:
        param.Hardware.num_workers = num_workers_override
    
    torch.set_num_threads(param.Hardware.torch_threads)

    # training run params after filling the defaults
    param_used_save = expt_dir/ Path('training_run_used').with_suffix(param_file.suffix)
    save_params(param_used_save, param)

    # setup tensorboard logger to write files to the directory appropriately
    log_dir_path = Path(param.Save.directory).parent / log_dir
    if not log_dir_path.exists():
        log_dir_path.mkdir(exist_ok=False)
    logger = SummaryWriter(log_dir=log_dir_path/expt_id)

    # save_test_dir
    save_test_dir = Path(param.Datasets.test.save_directory) / Path(expt_id)
    if not save_test_dir.exists():
        save_test_dir.mkdir(exist_ok=False)

    print(param)
    

    if param.Datasets.type == 'unet_dual':
        dataset_transformations = UnetTrainTransforms()
        dataset = MMDatasetUnetDual(data_dir= param.Datasets.directory, 
                                    transform=dataset_transformations, 
                                    weights=param.Datasets.weights)
    else:
        dataset = None
    
    len_dataset = len(dataset)
    train_len = int(len_dataset * param.Datasets.train.percentage)
    val_len = len_dataset - train_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    if param.Datasets.test.transformations.type == 'UnetTestTransforms':
        test_dataset_transformations = UnetTestTransforms()
        test_ds = MMDatasetUnetTest(images_dir=param.Datasets.test.directory, 
                                    transform=test_dataset_transformations)
    else:
        test_ds = None

    train_dl, val_dl, test_dl = setup_dataloader(param, train_ds, val_ds, test_ds)

    print(f"Train dataset length: {len(train_ds)} -- val: {len(val_ds)} -- test: {len(test_ds)}")
    # setup models and optimizers, loss functions, etc
    #print(model)
    model, optimizer, criterion, lr_scheduler = setup_trainer(param, device)

    # train loop
    nEpochs = param.HyperParameters.epochs
    print(f"Model training for epochs: {nEpochs}")
    print("\n --- Training cell and/or channel detetion model ---")
    current_lr = lr_scheduler.get_last_lr()[0]
    logger.add_scalar(param.HyperParameters.optimizer.name + '/lr', current_lr, 0)

    best_val_loss = 1_000_000
    for epoch in range(1, nEpochs + 1):
        model.train()
        epoch_loss = []
        for batch_i, (phase, mask, channel_mask, 
                    weights, filenames, raw_shapes) in enumerate(tqdm(train_dl, desc=f"Training epoch {epoch}")):
            optimizer.zero_grad()
            batches_done = len(train_dl) * epoch + batch_i
            phase_d = phase.to(device=device)
            if not isinstance(mask, list): # basically checking for nones
                mask_d = mask.to(device=device)
            else:
                mask_d = None
            if not isinstance(channel_mask, list):
                channel_mask_d = channel_mask.to(device=device)
            else:
                channel_mask_d = None
            
            if not isinstance(weights, list):
                weights_d = weights.to(device)
            else:
                weights_d = None

            predictions = model(phase_d)
            loss, loss_dict = criterion(predictions, mask_d, channel_mask_d, weights_d)
            loss.backward()
            epoch_loss.append(loss.item())
            optimizer.step()
            logger.add_scalar_dict('train/', loss_dict, global_step=batches_done)

        lr_scheduler.step()
        current_lr = lr_scheduler.get_last_lr()[0]
        logger.add_scalar(param.HyperParameters.optimizer.name + '/lr', current_lr, batches_done)
        print(f"Epoch: {epoch} Train loss: {np.mean(epoch_loss): 0.4f} lr: {current_lr : .6f}")

        del predictions
        torch.cuda.empty_cache()
        # validation loop
        model.eval()
        epoch_val_loss = []
        for batch_i, (phase, mask, channel_mask,
                    weights, filenames, raw_shapes) in enumerate(tqdm(val_dl, desc=f"Validation Epoch {epoch}")):
            
            batches_val_done = len(val_dl) * epoch + batch_i
            phase_d = phase.to(device=device)
            if not isinstance(mask, list): # basically checking for nones
                mask_d = mask.to(device=device)
            else:
                mask_d = None
            if not isinstance(channel_mask, list):
                channel_mask_d = channel_mask.to(device=device)
            else:
                channel_mask_d = None
            
            if not isinstance(weights, list):
                weights_d = weights.to(device)
            else:
                weights_d = None

            predictions = model(phase_d)
            loss_val, loss_val_dict = criterion(predictions, mask_d, channel_mask_d, weights_d)
            epoch_val_loss.append(loss_val.item())
            logger.add_scalar_dict('validation/', loss_val_dict, global_step=batches_val_done)
        mean_epoch_val_loss = np.mean(epoch_val_loss) 
        print(f"Epoch: {epoch} Validation loss: {mean_epoch_val_loss: .4f}")

        del predictions
        torch.cuda.empty_cache()
        # save model if 
        if mean_epoch_val_loss < best_val_loss:
            best_val_loss = mean_epoch_val_loss
            torch.save(model.state_dict(), model_out)
        
        # Test loop
        model.eval()
        for batch_i, (phase, filenames, raw_shapes) in enumerate(tqdm(test_dl, desc=f"Test Epoch {epoch}")):
            batches_test_done = len(test_dl) * epoch + batch_i
            #print(filenames)
            with torch.no_grad():
                predictions = model(phase.to(device, non_blocking=True))
                predictions = predictions.sigmoid() > param.Datasets.test.probability
                # log some images to monitor progress
                #if batch_i % 100 == 0:
                batch_fig_handles = plot_results_batch(to_cpu(phase).numpy(), to_cpu(predictions).numpy())
                for i, figure in enumerate(batch_fig_handles, 0):
                    logger.add_figure('test/fig' + str(batch_i), figure, global_step=batches_test_done)
                    plt.close(figure)
                # on last epoch save figures to directory
                #if epoch == nEpochs:
                #    batch_fig_handles == plot_results_batch(to_cpu(phase).numpy(), to_cpu(predictions).numpy())
                #    for i, figure in enumerate(batch_fig_handles, 0):
                #        save_path = save_test_dir / Path(filenames[i])
                #        figure.savefig(save_path, bbox_inches='tight')
                #        plt.close(figure)
            del predictions
            torch.cuda.empty_cache()

    print("\n ---- Training done ----\n")


def setup_trainer(param, device):
    """

    Returns:
        model: 
        optimizer:
        criterion:
        lr_scheduler:
    """
    model = model_dict[param.HyperParameters.architecture]
    model = model.parse(channels_by_scale=param.HyperParameters.model_params.channels_by_scale,
                        num_outputs=param.HyperParameters.model_params.num_outputs,
                        upsample_type=param.HyperParameters.model_params.upsample_type,
                        feature_fusion_type=param.HyperParameters.model_params.feature_fusion_type).to(device=device)
    
    if param.HyperParameters.loss == "unet_dual":
        #print("loss function set to dual loss")
        criterion = UnetDualLoss()

    if param.HyperParameters.optimizer.name == 'AdamW' :
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

    return model, optimizer, criterion, lr_scheduler
    

def setup_dataloader(param, train_ds, val_ds=None, test_ds=None):
    """
    Setup dataloaders for the datasets
    Args:
        param (RecursiveNamespace): parameter namespace
        train_ds (torch.utils.Dataset): an instance of Dataset class
        val_ds (optional): an instance of Dataset class
        test_ds (optional): an instance of Dataset class
    Return:
        train_dl: dataloader 
        val_dl: dataloader
        test_dl: dataloader
    """
    train_dl = DataLoader(
        dataset=train_ds,
        batch_size=param.HyperParameters.train_batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=param.Hardware.num_workers,
        pin_memory=False,
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
            drop_last=False,
            shuffle=False,
            num_workers=param.Hardware.num_workers,
            pin_memory=False,
            collate_fn=test_ds.collate_fn
        )
    else:
        test_dl = None
    
    return train_dl, val_dl, test_dl


def main():
    print("Hello from segmentation training")
    tprint("SEGMENT")
    args = parse_args()

    train_model(args.param_file,
                device_overwrite=args.device,
                num_workers_override=args.num_workers_override,
                log_dir=args.log_dir,
                log_comment=args.log_comment,
            )

if __name__ == "__main__":
    main()