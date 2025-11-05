import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from p2pnet.model import P2PNet
from p2pnet.dataset import UnderwaterDataset, get_transforms, get_eval_transforms
from p2pnet.metrics import (
    LossSuite,
    calculate_psnr,
    calculate_ssim,
    calculate_uciqe,
    calculate_uiqm,
)
from p2pnet.utils import load_config, save_checkpoint, load_checkpoint, save_sample_images
from PIL import Image

def main(args):
    # --- Setup ---
    config = load_config(args.config)

    # Device
    if config['train']['device'] == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config['train']['device'])

    print(f"Using device: {device}")

    # Adjust batch size for CPU
    if device.type == 'cpu' and config['train']['batch_size'] > 4:
        print(f"Reducing batch size from {config['train']['batch_size']} to 4 for CPU.")
        config['train']['batch_size'] = 4

    # Create directories
    os.makedirs(config['log']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['log']['output_dir'], exist_ok=True)

    # --- Data ---
    train_transform = get_transforms(config)
    val_transform = get_eval_transforms(config['model']['input_size'])

    train_dataset = UnderwaterDataset(root_dir=config['data']['train_path'], transform=train_transform)
    val_dataset = UnderwaterDataset(root_dir=config['data']['val_path'], transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )

    # --- Model, Optimizer, Loss ---
    model = P2PNet(base_ch=config['model']['base_channels']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['train']['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['train']['epochs'], eta_min=1e-6)

    loss_suite = LossSuite(config['loss_weights'], device=device)

    # --- Resuming ---
    start_epoch = 0
    best_uciqe = 0.0
    checkpoint_path = args.checkpoint if args.checkpoint else config['train']['resume_checkpoint']
    if checkpoint_path and os.path.exists(checkpoint_path):
        start_epoch, best_uciqe = load_checkpoint(checkpoint_path, model, optimizer, scheduler)
        print(f"Resuming training from epoch {start_epoch}. Best UCIQE: {best_uciqe:.4f}")

    # --- Training Loop ---
    for epoch in range(start_epoch, config['train']['epochs']):
        model.train()
        epoch_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['train']['epochs']}", leave=True)
        for raw_imgs, gt_imgs in progress_bar:
            raw_imgs, gt_imgs = raw_imgs.to(device), gt_imgs.to(device)

            optimizer.zero_grad()
            pred_imgs = model(raw_imgs)

            total_loss = loss_suite(pred_imgs, gt_imgs)

            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            progress_bar.set_postfix(loss=f'{total_loss.item():.4f}')

        scheduler.step()
        avg_epoch_loss = epoch_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_psnr, val_ssim, val_uciqe, val_uiqm = 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            for i, (raw_imgs, gt_imgs) in enumerate(val_loader):
                raw_imgs, gt_imgs = raw_imgs.to(device), gt_imgs.to(device)
                pred_imgs = model(raw_imgs)

                # Convert to numpy for metrics
                pred_np = pred_imgs.permute(0, 2, 3, 1).cpu().numpy()
                gt_np = gt_imgs.permute(0, 2, 3, 1).cpu().numpy()
                raw_np = raw_imgs.permute(0, 2, 3, 1).cpu().numpy()

                for j in range(pred_np.shape[0]):
                    val_psnr += calculate_psnr(pred_np[j], gt_np[j])
                    val_ssim += calculate_ssim(pred_np[j], gt_np[j])
                    val_uciqe += calculate_uciqe(pred_np[j])
                    val_uiqm += calculate_uiqm(pred_np[j])

                # Save one sample batch from validation
                if i == 0 and (epoch + 1) % config['log']['sample_every_epochs'] == 0:
                    save_path = os.path.join(config['log']['output_dir'], f"epoch_{epoch+1}_sample.png")
                    save_sample_images(raw_np, pred_np, gt_np, save_path)

        # Average metrics
        num_val_images = len(val_dataset)
        avg_psnr = val_psnr / num_val_images
        avg_ssim = val_ssim / num_val_images
        avg_uciqe = val_uciqe / num_val_images
        avg_uiqm = val_uiqm / num_val_images

        print(f"Epoch {epoch+1} Summary | Loss: {avg_epoch_loss:.4f} | "
              f"PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.4f} | "
              f"UCIQE: {avg_uciqe:.4f} | UIQM: {avg_uiqm:.4f}")

        # --- Checkpointing ---
        if (epoch + 1) % config['log']['save_every_epochs'] == 0:
            save_checkpoint(
                epoch + 1, model, optimizer, scheduler, best_uciqe,
                os.path.join(config['log']['checkpoint_dir'], f"checkpoint_epoch_{epoch+1}.pth")
            )

        if avg_uciqe > best_uciqe:
            best_uciqe = avg_uciqe
            print(f"New best UCIQE: {best_uciqe:.4f}. Saving best model.")
            save_checkpoint(
                epoch + 1, model, optimizer, scheduler, best_uciqe,
                os.path.join(config['log']['checkpoint_dir'], "best_model.pth")
            )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train P2PNet for Underwater Image Enhancement")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file (.yaml or .json)')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to resume training')
    parser.add_argument('--device', type=str, default=None, help='Force device (e.g., "cpu" or "cuda")')

    args = parser.parse_args()
    if args.device:
        # Override config device if CLI flag is set
        config = load_config(args.config)
        config['train']['device'] = args.device

    main(args)