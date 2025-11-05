# Technical Summary of P2PNet

This document provides a technical overview of the P2PNet implementation for underwater image enhancement.

## 1. Model Architecture

The model is a fully convolutional U-Net-like network that operates without any spatial downsampling, preserving the `256x256` resolution throughout the network.

### Encoder
The encoder consists of three convolutional blocks. A key innovation is the integration of a **Channel Attention (CA) module** in each block.

* **Input**: An RGB image `Fin` of size `(B, 3, 256, 256)`.
* **Block `i` Operation**:
    1.  A standard convolutional layer (`Conv2d -> BatchNorm2d -> ReLU`) processes the input feature map `F_in`.
    2.  The resulting feature map `F_out` undergoes **Global Pooling**, where both Global Average Pooling and Global Max Pooling are computed and added together.
    3.  This pooled tensor is passed through a simple two-layer MLP (the attention mechanism) with a final `Sigmoid` activation to produce channel-wise weights `w`.
    4.  The original feature map `F_out` is scaled by these weights: `F_CA = F_out * w`.
* **Skip Connections**: The channel-attended feature maps (`F_CA`) are used for skip connections to the decoder.

### Decoder
The decoder mirrors the encoder structure, using `ConvTranspose2d` layers for up-sampling (though in this architecture, they function as regular convolutions since spatial resolution is constant).

* **Block `i` Operation**:
    1.  The feature map from the previous decoder layer is passed through a `ConvTranspose2d` layer.
    2.  The output is concatenated with the corresponding channel-attended skip connection from the encoder.
    3.  A final convolutional layer refines the concatenated features.
* **Output**: The final layer uses a `Conv2d` followed by a `Sigmoid` activation to produce the enhanced RGB image `Y`, with pixel values in the range `[0, 1]`.

## 2. Loss Function

The total loss is a weighted sum of three components, designed to balance pixel-level accuracy, perceptual quality, and color distribution.

$L_{total} = L_1 + \lambda_{hist} \cdot L_{hist} + \lambda_{ssim} \cdot L_{ssim}$

Where:
* $L_1$: Standard L1 pixel-wise loss (`torch.nn.L1Loss`).
* $L_{hist}$: A differentiable **Soft Gaussian Histogram Loss**. It encourages the color histogram of the predicted image to match that of the ground truth. It is implemented with 64 bins and a sigma of 0.02.
* $L_{ssim}$: Structural Similarity Index Measure loss, penalizing perceptual differences.
* **Weights**: $\lambda_{hist} = 0.1$, $\lambda_{ssim} = 0.05$.

## 3. Training and Evaluation

* **Optimizer**: Adam (`lr=1e-4`).
* **Scheduler**: `CosineAnnealingLR` for smooth learning rate decay.
* **Epochs**: 100.
* **Batch Size**: 8 (with automatic reduction for CPU execution to prevent memory issues).
* **Checkpointing**:
    * A checkpoint is saved every 5 epochs.
    * The best-performing model is saved as `checkpoints/best_model.pth` based on the validation set's **UCIQE** score, as it is a reliable no-reference metric for underwater image colorfulness and contrast.
* **Metrics Logged**:
    * **Loss**: Total training loss.
    * **PSNR**: Peak Signal-to-Noise Ratio.
    * **SSIM**: Structural Similarity Index Measure.
    * **UCIQE**: Underwater Color Image Quality Evaluation.
    * **UIQM**: Underwater Image Quality Measure.

## 4. Inference and Deployment

The `infer.py` script is designed for easy use:
* It accepts a path to a trained model's weights (`.pth` file).
* It can process either a single image or an entire directory of images.
* It handles device placement (CPU/GPU) automatically.