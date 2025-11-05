import os
import argparse
import glob
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

from p2pnet.model import P2PNet
from p2pnet.utils import load_config

def infer(args):
    # --- Setup ---
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(args.output, exist_ok=True)

    # --- Model ---
    model = P2PNet(base_ch=args.base_ch).to(device)

    try:
        checkpoint = torch.load(args.weights, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded weights from {args.weights}")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    model.eval()

    # --- Data ---
    transform = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor()
    ])

    if os.path.isdir(args.input):
        image_paths = glob.glob(os.path.join(args.input, "*.[pP][nN][gG]")) + \
                      glob.glob(os.path.join(args.input, "*.[jJ][pP][gG]")) + \
                      glob.glob(os.path.join(args.input, "*.[jJ][pP][eE][gG]"))
    else:
        image_paths = [args.input]

    if not image_paths:
        print(f"No images found in {args.input}")
        return

    # --- Inference Loop ---
    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="Processing images"):
            try:
                img = Image.open(img_path).convert("RGB")
                input_tensor = transform(img).unsqueeze(0).to(device)

                output_tensor = model(input_tensor)

                output_img_np = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                output_img_np = (output_img_np * 255).astype(np.uint8)

                output_img = Image.fromarray(output_img_np)

                basename = os.path.basename(img_path)
                name, ext = os.path.splitext(basename)
                save_path = os.path.join(args.output, f"{name}_out{ext}")

                output_img.save(save_path)
            except Exception as e:
                print(f"Failed to process {img_path}: {e}")

    print(f"Inference complete. Results saved in {args.output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference script for P2PNet")
    parser.add_argument('--weights', type=str, required=True, help='Path to the trained model weights (.pth)')
    parser.add_argument('--input', type=str, required=True, help='Path to an input image or a folder of images')
    parser.add_argument('--output', type=str, default='inference_results', help='Path to the output directory')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use for inference')
    parser.add_argument('--base_ch', type=int, default=32, help='Base channels of the model (must match trained model)')
    parser.add_argument('--size', type=int, default=256, help='Image size for inference')

    args = parser.parse_args()
    infer(args)