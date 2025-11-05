import os
import numpy as np
from PIL import Image

def create_dummy_image(path, size=(256, 256)):
    """Creates a random noise image."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    array = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
    img = Image.fromarray(array)
    img.save(path)

def main():
    """Generates a dummy dataset for testing the pipeline."""
    base_dir = "data"
    test_dir = "test_images"

    # Structure: {split: {folder: num_images}}
    structure = {
        "train": {"raw": 800, "gt": 800},
        "val": {"raw": 100, "gt": 100},
    }

    print("Generating dummy dataset...")
    for split, folders in structure.items():
        for folder, count in folders.items():
            dir_path = os.path.join(base_dir, split, folder)
            for i in range(count):
                img_path = os.path.join(dir_path, f"dummy_{i:04d}.png")
                create_dummy_image(img_path)
            print(f"Generated {count} images in {dir_path}")

    print("\nGenerating dummy test images...")
    os.makedirs(test_dir, exist_ok=True)
    for i in range(10):
        img_path = os.path.join(test_dir, f"test_image_{i}.png")
        create_dummy_image(img_path)
    print(f"Generated 10 images in {test_dir}")

    print("\nDummy data generation complete.")

if __name__ == "__main__":
    main()