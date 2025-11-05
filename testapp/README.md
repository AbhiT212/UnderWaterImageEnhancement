# P2PNet for Underwater Image Enhancement

This repository contains the official PyTorch implementation of **P2PNet**, a deep learning model designed for enhancing underwater images. The model architecture features a U-Net-like encoder-decoder structure with a novel **Global Pooling + Channel Attention** mechanism in the encoder path to effectively learn and restore features degraded by underwater conditions.



## Features

* **Channel Attention Mechanism**: Encoder blocks use Global Average and Max Pooling followed by Channel Attention to refine feature maps.
* **Comprehensive Loss Function**: A combination of L1, SSIM, and a differentiable Histogram Loss ensures perceptual quality and color correction.
* **Full Training & Inference Pipeline**: Includes scripts for training, inference, and evaluation.
* **Underwater-Specific Metrics**: Implements and tracks **UCIQE** and **UIQM** for robust performance evaluation.
* **Colab Ready**: A pre-configured notebook (`colab_notebook.ipynb`) allows for quick setup and execution on Google Colab.
* **Reproducibility**: Comes with configuration files (`config.yaml`, `config.json`) to ensure experiments are reproducible.

## Directory Structure

```
p2pnet-underwater/
├── README.md
├── SUMMARY.md
├── requirements.txt
├── colab_notebook.ipynb
├── configs/
│   ├── config.yaml
│   └── config.json
├── scripts/
│   └── generate_dummy_data.py
├── data/
├── test_images/
├── p2pnet/
│   ├── __init__.py
│   ├── model.py
│   ├── dataset.py
│   ├── train.py
│   ├── infer.py
│   ├── metrics.py
│   └── utils.py
├── checkpoints/
├── outputs/
└── tests/
    ├── test_dataset_pairing.py
    └── test_forward_pass.py
```

## Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd p2pnet-underwater
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Data

Place your training and validation images in the `data/` directory following the structure below. You can also use the dummy data generator.

```bash
# To generate placeholder images for testing
python scripts/generate_dummy_data.py
```

**Data Structure:**
```
data/
├── train/
│   ├── raw/      # Raw underwater images
│   └── gt/       # Ground truth clear images
└── val/
    ├── raw/
    └── gt/
```

## Usage

### Training

Train the model using the provided configuration. The script will automatically use a GPU if available.

```bash
python -m p2pnet.train --config configs/config.yaml
```

**Key Arguments:**
* `--config`: Path to the configuration file.
* `--checkpoint`: (Optional) Path to a `.pth` file to resume training.
* `--device`: (Optional) Manually set device (`cuda` or `cpu`).

Checkpoints will be saved in `checkpoints/` and sample outputs in `outputs/`. The best model is saved as `best_model.pth` based on the highest UCIQE score.

### Inference

Run inference on a single image or a folder of images.

```bash
# Inference on a single image
python -m p2pnet.infer --weights checkpoints/best_model.pth --input /path/to/image.jpg --output /path/to/save/

# Inference on a folder
python -m p2pnet.infer --weights checkpoints/best_model.pth --input /path/to/folder/ --output /path/to/save_folder/
```

## Testing

Run unit tests to verify the dataset pairing and model forward pass.

```bash
python -m unittest discover tests
```

## Acknowledgements

This implementation is based on foundational concepts from U-Net and attention mechanisms in deep learning. We also acknowledge the authors of the UIQM and UCIQE metrics.