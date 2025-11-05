import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class UnderwaterDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.raw_path = os.path.join(root_dir, 'raw')
        self.gt_path = os.path.join(root_dir, 'gt')

        self.raw_images = sorted(glob.glob(os.path.join(self.raw_path, "*")))
        self.gt_images = sorted(glob.glob(os.path.join(self.gt_path, "*")))

        self._validate_and_pair()

    def _validate_and_pair(self):
        """
        Validates dataset and creates pairs.
        First attempts to pair by basename, then falls back to sorted order.
        """
        if not self.raw_images or not self.gt_images:
            raise ValueError(f"No images found in {self.root_dir}")

        if len(self.raw_images) != len(self.gt_images):
            print(f"Warning: Mismatched number of raw ({len(self.raw_images)}) and gt ({len(self.gt_images)}) images.")
            # Fallback to pairing by sorted lists, trimming to the shorter list
            min_len = min(len(self.raw_images), len(self.gt_images))
            self.raw_images = self.raw_images[:min_len]
            self.gt_images = self.gt_images[:min_len]
            print(f"Using {min_len} paired images based on sorted order.")
            return

        # Try to pair by basename
        gt_map = {os.path.basename(p): p for p in self.gt_images}
        paired_raw = []
        paired_gt = []
        unpaired_count = 0

        for raw_p in self.raw_images:
            basename = os.path.basename(raw_p)
            if basename in gt_map:
                paired_raw.append(raw_p)
                paired_gt.append(gt_map[basename])
            else:
                unpaired_count += 1

        if unpaired_count > 0:
            print(f"Warning: {unpaired_count} images could not be paired by name. Falling back to sorted order for all images.")
            # If any are unpaired, fall back to simple sorted list pairing for consistency
            self.image_pairs = list(zip(sorted(self.raw_images), sorted(self.gt_images)))
        else:
            print("Successfully paired all images by filename.")
            self.image_pairs = list(zip(paired_raw, paired_gt))

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        raw_path, gt_path = self.image_pairs[idx]

        raw_image = Image.open(raw_path).convert("RGB")
        gt_image = Image.open(gt_path).convert("RGB")

        if self.transform:
            # Apply same random transform to both images
            # To ensure flips/rotations are identical, we get parameters first
            state = torch.get_rng_state()
            raw_image = self.transform(raw_image)
            torch.set_rng_state(state)
            gt_image = self.transform(gt_image)

        return raw_image, gt_image

def get_transforms(config):
    """Returns a torchvision transform object based on the config."""
    aug_cfg = config['augmentations']
    transform = transforms.Compose([
        transforms.Resize((config['model']['input_size'], config['model']['input_size'])),
        transforms.RandomHorizontalFlip(p=aug_cfg['hflip_prob']),
        transforms.RandomVerticalFlip(p=aug_cfg['vflip_prob']),
        transforms.RandomRotation(aug_cfg['rotation_degrees']),
        transforms.ColorJitter(
            brightness=aug_cfg['color_jitter_brightness'],
            contrast=aug_cfg['color_jitter_contrast']
        ),
        transforms.ToTensor() # Converts to [0, 1] range
    ])
    return transform

def get_eval_transforms(input_size=256):
    """Transforms for validation/testing (no augmentation)."""
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor()
    ])
    return transform