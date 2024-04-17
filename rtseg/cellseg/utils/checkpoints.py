
import torch
import pathlib

def save_checkpoint(filename, checkpoint):
    """
    Save checkpoint which is a dictionary 
        state = {
            "optimizer_state_dict": optimizer.state_dict(),
            "model_state_dict" : model.state_dict(),
            "epoch": epoch
        }
    to filename

    Args:
        filename: path to save the checkpoint dict
        checkpoint: a state dictionary with containig the above described keys and values
    Returns:
        None
    """
    filename = filename if isinstance(filename, pathlib.Path) else pathlib.Path(filename)
    if not filename.parent.exists():
        filename.parent.mkdir(exist_ok=True, parents=True) 

    #print(f"Saving checkpoing to : {str(filename)}")
    torch.save(checkpoint, filename)
    



def load_checkpoint(filename, model, optimizer = None):
    """
    Load state = {
             "optimizer_state_dict": optimizer.state_dict(),
            "model_state_dict" : model.state_dict(),
            "epoch": epoch
        }
    from filename and initializes the model and optimizer
    You are on your own to load the correct model and optimzier,
    there are not checks yet. The code will most likely due to mismatches.
    
    Args:
        filename:
        model: a torch.nn module to load state dict into
        optimizer: also and torch.optim module to load state dict into

    """
    filename = filename if isinstance(filename, pathlib.Path) else pathlib.Path(filename)
    assert filename.exists(), "Checkpoint file doesn't exist" 

    print(f"Loading checkpoint from file: {str(filename)}")
    checkpoint = torch.load(filename, map_location=torch.device("cpu"))

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint["epoch"]

    return model, optimizer, epoch