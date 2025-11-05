import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_ski
from skimage.metrics import structural_similarity as ssim_ski
from scipy.signal import convolve2d
from PIL import Image

# --- Loss Functions ---

class SSIMLoss(nn.Module):
    """Differentiable SSIM loss"""
    def ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2
        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
        C1 = 0.01**2
        C2 = 0.03**2
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        window_size = 11
        sigma = 1.5
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        gauss = gauss/gauss.sum()
        _1D_window = gauss.unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous().to(img1.device)

        return 1.0 - self.ssim(img1, img2, window, window_size, channel)

class HistLoss(nn.Module):
    """Differentiable Soft Histogram Loss"""
    def __init__(self, bins=64, sigma=0.02, device='cpu'):
        super(HistLoss, self).__init__()
        self.bins = bins
        self.sigma = sigma
        self.bin_centers = torch.linspace(0, 1, bins).view(1, -1, 1, 1).to(device)

    def get_soft_hist(self, x):
        x = x.unsqueeze(1)
        pdf = torch.exp(-((x - self.bin_centers)**2) / (2 * self.sigma**2))
        hist = pdf.mean(dim=[2, 3])
        return hist / hist.sum(dim=1, keepdim=True)

    def forward(self, pred, target):
        loss = 0.0
        for i in range(pred.shape[1]):
            pred_hist = self.get_soft_hist(pred[:, i, :, :].unsqueeze(1))
            target_hist = self.get_soft_hist(target[:, i, :, :].unsqueeze(1))
            loss += torch.abs(pred_hist - target_hist).sum(dim=1).mean()
        return loss / pred.shape[1]

class LossSuite(nn.Module):
    """Combines all losses into one module."""
    def __init__(self, weights, device):
        super(LossSuite, self).__init__()
        self.weights = weights
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss()
        self.hist_loss = HistLoss(device=device)

    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        hist = self.hist_loss(pred, target)

        total_loss = (self.weights['l1'] * l1 + 
                      self.weights['ssim'] * ssim + 
                      self.weights['hist'] * hist)
        return total_loss

# --- Evaluation Metrics (Numpy based) ---

def calculate_psnr(pred, gt):
    return psnr_ski(gt, pred, data_range=1.0)

def calculate_ssim(pred, gt):
    return ssim_ski(gt, pred, data_range=1.0, channel_axis=2)

def calculate_uciqe(img_np):
    """
    Calculate UCIQE (Underwater Color Image Quality Evaluation).
    Input: numpy image in [0, 1], shape (H, W, C)
    """
    try:
        img_lab = np.array(Image.fromarray((img_np * 255).astype(np.uint8)).convert('LAB')).astype(float)
        c1, c2, c3 = 0.4680, 0.2745, 0.2576

        l_channel = img_lab[:, :, 0]
        a_channel = img_lab[:, :, 1]
        b_channel = img_lab[:, :, 2]

        chroma = np.sqrt(a_channel**2 + b_channel**2)
        sigma_c = np.std(chroma)

        saturation = np.sqrt((l_channel - np.mean(l_channel))**2 + chroma**2)
        mu_s = np.mean(saturation)

        contrast_l = np.max(l_channel) - np.min(l_channel)

        return c1 * sigma_c + c2 * contrast_l + c3 * mu_s
    except Exception:
        return 0.0

def calculate_uiqm(img_np):
    """
    Calculate UIQM (Underwater Image Quality Measure).
    Input: numpy image in [0, 1], shape (H, W, C)
    """
    try:
        img_rgb = (img_np * 255).astype(np.uint8)
        c1, c2, c3 = 0.0282, 0.2953, 3.5753

        RG = img_rgb[:, :, 0].astype(float) - img_rgb[:, :, 1].astype(float)
        YB = (img_rgb[:, :, 0].astype(float) + img_rgb[:, :, 1].astype(float)) / 2 - img_rgb[:, :, 2].astype(float)
        uicm = np.sqrt(np.mean(RG**2) + np.mean(YB**2))

        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = sobel_x.T

        g_r = convolve2d(img_rgb[:, :, 0], sobel_x, mode='same') + convolve2d(img_rgb[:, :, 0], sobel_y, mode='same')
        g_g = convolve2d(img_rgb[:, :, 1], sobel_x, mode='same') + convolve2d(img_rgb[:, :, 1], sobel_y, mode='same')
        g_b = convolve2d(img_rgb[:, :, 2], sobel_x, mode='same') + convolve2d(img_rgb[:, :, 2], sobel_y, mode='same')
        uism = 0.5 * np.log(np.mean(g_r**2)) + 0.5 * np.log(np.mean(g_g**2)) + 0.5 * np.log(np.mean(g_b**2))

        eme, h, w, _ = 0, *img_rgb.shape
        block_size = 8
        num_blocks = 0
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                block = img_rgb[i:i+block_size, j:j+block_size].astype(float)
                if block.size == 0: continue
                max_val, min_val = np.max(block), np.min(block)
                if min_val > 0:
                    eme += np.log(max_val / min_val)
                num_blocks += 1
        uiconm = eme / num_blocks if num_blocks > 0 else 0

        return c1 * uicm + c2 * uism + c3 * uiconm
    except Exception:
        return 0.0