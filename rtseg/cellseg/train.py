
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from rtseg.utils.param_io import load_params, save_params
import yaml
from rtseg.cellseg.networks import model_dict
from rtseg.cellseg.utils.checkpoints import load_checkpoint, save_checkpoint
from rtseg.cellseg.dataloaders import PhaseContrast
from rtseg.cellseg.utils.transforms import all_transforms
from rtseg.cellseg.losses import IVPLoss, TverskyLoss
from torch.utils.data import DataLoader, random_split
from rtseg.utils.logger import SummaryWriter
import torch
import torch.nn as nn

def parse_args():
    parser = argparse.ArgumentParser(description="Segmentation network training config file")
    parser.add_argument('-p', '--param_file',
                        help='Specify parameter file for training segmentation network',
                        required=True)

    parser.add_argument('-l', '--log_dir', default='',
                        help='Set directory where you want the tensorboard logs to go in',
                        type=str, required=False)
    
    parser.add_argument('-c', '--log_comment', default='',
                        help="Add a comment to the saving directory",
                        type=str, required=False)

    args = parser.parse_args()

    return args

def train_model(param_file: str | Path, log_dir: str | Path = '', 
                log_comment: str = ''):
    """
    Train a segmentation model using the parameter give in the 
    parameter file.

    Args:
        param_file (str): parameter file path, that includes everything
            you need to train segmentation network
            (check rtseg/cellseg/configs/cellseg.yaml for reference)
        
        log_dir (str): tensorboard logs dir so that you can compare different models

        log_comment (str): 
    
    """
    param_file = Path(param_file)
    if not param_file.exists():
        raise FileNotFoundError(f"Parameter file {str(param_file)} doesn't exist")

    param = load_params(param_file, 'cellseg') 

    # Create directory for storing parameters and model files
    expt_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if log_comment:
        expt_id = expt_id + '_' + log_comment

    # create a directory, every time you run the train model function
    expt_dir = Path(param.Save.directory) / Path(expt_id)
    expt_dir.mkdir(parents=True, exist_ok=False)

    # Logger
    if not log_dir:
        log_dir = Path(param.Save.log_dir)
    else:
        log_dir = Path(log_dir)

    if not log_dir.exists():
        log_dir.mkdir(exist_ok=False)
    logger = SummaryWriter(log_dir=log_dir / Path(expt_id))

    print("---- Parameters set ----")
    print(yaml.dump(param.to_dict()))
    print("------------------------")

    # save parameters to expt dict
    save_params(expt_dir / Path('training_params.yaml'), param)

    # device to train
    device = torch.device(param.Hardware.device)

    print(f"Experiment: {expt_id} is training on device: {device}")

    # Datasets
    train_dir = Path(param.Dataset.Train.directory)
    train_transforms = all_transforms[param.Dataset.Train.transforms]
    train_dataset = PhaseContrast(phase_dir=train_dir / Path(param.Dataset.Train.phase_dir),
                    labels_dir=train_dir/Path(param.Dataset.Train.labels_dir),
                    vf_dir=train_dir/Path(param.Dataset.Train.vf_dir),
                    vf_at_runtime=param.Dataset.Train.vf_at_runtime,
                    labels_delimiter=param.Dataset.Train.labels_delimiter,
                    vf_delimiter=param.Dataset.Train.vf_delimiter,
                    transforms=train_transforms,
                    phase_format=param.Dataset.Train.phase_format,
                    labels_format=param.Dataset.Train.labels_format,
                    vf_format=param.Dataset.Train.vf_format
                )
    len_train_dataset = len(train_dataset)
    val_len = int(len_train_dataset * param.Dataset.Train.validation_percentage)

    train_ds, val_ds = random_split(train_dataset, [len_train_dataset - val_len, val_len])

    # Dataset on which evals are done
    #test_dir = Path(param.Dataset.Test.directory)
    #test_transforms = transforms[param.Dataset.Test.transforms]
    #test_ds = PhaseContrast(phase_dir=test_dir/Path(param.Dataset.Test.phase_dir),
    #                labels_dir=test_dir/Path(param.Dataset.Test.labels_dir),
    #                vf_dir=test_dir/Path(param.Dataset.Test.vf_dir),
    #                vf=param.Dataset.Test.vf,
    #                labels_delimiter=param.Dataset.Test.labels_delimiter,
    #                vf_delimiter=param.Dataset.Test.vf_delimiter,
    #                transforms=test_transforms,
    #                phase_format=param.Dataset.Test.phase_format,
    #                labels_format=param.Dataset.Test.labels_format,
    #                vf_format=param.Dataset.Test.vf_format
    #            )

    train_dl = DataLoader(train_ds, 
                          batch_size=param.Training.batch_size,
                          pin_memory=False,
                          shuffle=True,
                          drop_last=True,
                          num_workers=param.Training.num_workers)

    val_dl = DataLoader(val_ds,
                          batch_size=param.Training.val_batch_size,
                          pin_memory=False,
                          shuffle=False,
                          drop_last=False,
                          num_workers=param.Training.num_workers)
 
    #test_dl = DataLoader(test_ds, batch_size=param.Testing.batch_size,
    #                      pin_memory=False,
    #                      shuffle=False,
    #                      drop_last=False,
    #                      num_workers=param.Testing.num_workers)
 
    # Model
    model = model_dict[param.Training.model]
    model = model.parse(**param.Training.model_params.to_dict())

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=param.Training.optimizer.lr)  # type: ignore  

    # losses
    vf_losses = []
    semantic_losses = []

    if param.Training.losses.mse_loss.apply:
        vf_losses.append(["MSE loss", nn.MSELoss()])
    if param.Training.losses.ivp_loss.apply:
        vf_losses.append(["IVP loss", IVPLoss(
            dx=param.Training.losses.ivp_loss.dx,
            n_steps=param.Training.losses.ivp_loss.n_steps,
            solver=param.Training.losses.ivp_loss.solver,
            device=device 
        )])
    if param.Training.losses.tversky_loss.apply:
        semantic_losses.append(["Tversky loss",  TverskyLoss(
            alpha=param.Training.losses.tversky_loss.alpha,
            beta=param.Training.losses.tversky_loss.beta,
            from_logits=param.Training.losses.tversky_loss.from_logits
        )])
    
    if param.Training.losses.bce_loss.apply:
        semantic_losses.append(["BCE loss", nn.BCELoss()])

    # load the model, optimizer and epoch number from the check point file
    if param.Checkpoints.load is True:
        # load checkpoints
        model, optimizer, last_epoch = load_checkpoint(param.Checkpoints.filename, model, optimizer)
    else:
        last_epoch = 0
        
    # Train loop
    model.to(device) # type: ignore
    best_val_loss = 1_000_000
    for e in range(last_epoch, param.Training.nEpochs):
        model.train() # type: ignore
        running_train_loss = []
        for batch_i, (phase_batch, masks_batch, vf_batch) in enumerate(tqdm(train_dl, desc=f"Training epoch: {e}")):
            phase_batch = phase_batch.to(device)
            masks_batch = masks_batch.to(device)
            vf_batch = vf_batch.to(device) 
            batches_done = len(train_dl) * e + batch_i
            # masks are labels, convert them to boolean
            semantic = torch.where(masks_batch > 0, 1.0, 0.0)
            optimizer.zero_grad()

            # pass through model
            pred_semantic, pred_vf = model(phase_batch) # type: ignore
            # calculate loss
            loss = 0
            for name, loss_f in vf_losses:
                loss_value = loss_f(pred_vf, vf_batch) # type: ignore
                loss += loss_value
            for name, loss_f in semantic_losses:
                loss_value = loss_f(pred_semantic, semantic) # type: ignore
                loss += loss_value

            loss.backward() # type: ignore
            optimizer.step()
            # gather loss for reporting
            running_train_loss.append(loss.item()) # type: ignore

        running_train_loss_mean = np.mean(running_train_loss)
        logger.add_scalar_dict('train/', {'loss': running_train_loss_mean}, global_step=batches_done)

        if (e+1) % param.Training.save_checkpoints.every == 0:
            # save checkpoint file
            checkpoint = {
                "optimizer_state_dict" : optimizer.state_dict(), # type: ignore
                "model_state_dict" : model.state_dict(), # type: ignore
                "epoch": e
            }
            filename = Path(expt_dir) / Path('checkpoint_train.pt')
            save_checkpoint(filename, checkpoint)

        # --------------------------------------------
        # validation loop
        running_val_loss = []
        model.eval() # type: ignore
        with torch.no_grad():
            for batch_val, (phase_batch, masks_batch, vf_batch) in enumerate(tqdm(val_dl, desc=f"Validation epoch: {e}")):
                phase_batch = phase_batch.to(device)
                masks_batch = masks_batch.to(device)
                vf_batch = vf_batch.to(device) 
                semantic = torch.where(masks_batch > 0, 1.0, 0.0)
                batches_val_done = len(val_dl) * e + batch_val

                pred_semantic, pred_vf = model(phase_batch) # type: ignore

                val_loss = 0
                for name, loss_f in vf_losses:
                    loss_value = loss_f(pred_vf, vf_batch) # type: ignore
                    val_loss += loss_value
                for name, loss_f in semantic_losses:
                    loss_value = loss_f(pred_semantic, semantic) # type: ignore
                    val_loss += loss_value
                
                running_val_loss.append(val_loss.item()) # type: ignore

            running_val_loss_mean = np.mean(running_val_loss)
            logger.add_scalar_dict('validation/', {'val_loss': running_val_loss_mean}, global_step=batches_val_done)

        if running_val_loss_mean < best_val_loss:
            best_val_loss = running_val_loss_mean
            model_path = expt_dir / Path('model_val.pt')
            # write the model with the best val loss to disk
            torch.save(model.state_dict(), model_path) # type: ignore
            


        # --------------------------------------------
        #logger.add_scalar_dict("loss_cmp/", {'train': running_train_loss_mean,
        #                                                   'val': running_val_loss_mean}, e + 1)

def main():
    print("Segmentaiton training ....")
    args = parse_args()

    train_model(param_file=args.param_file,
                log_dir=args.log_dir,
                log_comment=args.log_comment)


if __name__ == '__main__':
    main()