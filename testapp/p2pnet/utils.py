import yaml
import json
import torch
import numpy as np
from PIL import Image

def load_config(config_path):
    """Loads a YAML or JSON configuration file."""
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    elif config_path.endswith('.json'):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError("Config file must be .yaml, .yml, or .json")

def save_checkpoint(epoch, model, optimizer, scheduler, best_metric, filepath):
    """Saves model checkpoint."""
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_metric': best_metric
    }
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(filepath, model, optimizer=None, scheduler=None):
    """Loads model checkpoint."""
    if not torch.cuda.is_available():
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(filepath)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    best_metric = checkpoint.get('best_metric', 0.0)

    return epoch, best_metric

def save_sample_images(raw, pred, gt, save_path):
    """
    Saves a concatenated image of (raw | pred | gt).
    Inputs are numpy arrays in [0, 1] range, shape (B, H, W, C).
    """
    # Use only the first image in the batch
    raw_img = (raw[0] * 255).astype(np.uint8)
    pred_img = (pred[0] * 255).astype(np.uint8)
    gt_img = (gt[0] * 255).astype(np.uint8)

    h, w, _ = raw_img.shape

    # Create a spacer
    spacer = np.ones((h, 10, 3), dtype=np.uint8) * 255 # White spacer

    concatenated_img = np.concatenate([raw_img, spacer, pred_img, spacer, gt_img], axis=1)

    Image.fromarray(concatenated_img).save(save_path)
    print(f"Sample image saved to {save_path}")