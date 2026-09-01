import os
import sys
import cv2
import pandas as pd

# ======================================
# Add project root to Python path
# ======================================

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from utils.brain_generator import BrainGenerator
from utils.tumor_generator import TumorGenerator
from utils.mask_generator import MaskGenerator

# ======================================
# Create Dataset Folders
# ======================================

IMAGE_FOLDER = "dataset/images"
MASK_FOLDER = "dataset/masks"

os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(MASK_FOLDER, exist_ok=True)

# ======================================
# Initialize Classes
# ======================================

brain_generator = BrainGenerator()
tumor_generator = TumorGenerator()
mask_generator = MaskGenerator()

# ======================================
# Dataset Settings
# ======================================

NUMBER_OF_IMAGES = 5000      # Change to 100 if testing

metadata = []

# ======================================
# Generate Dataset
# ======================================

for i in range(NUMBER_OF_IMAGES):

    # Step 1 : Generate Brain
    image = brain_generator.generate()

    # Step 2 : Add Tumor
    image, tumor_info = tumor_generator.draw(image)

    # Step 3 : Generate Segmentation Mask
    mask = mask_generator.create_mask(
        image_size=256,
        tumor_type=tumor_info["tumor_type"],
        center=tumor_info["center"],
        radius=tumor_info["radius"]
    )

    # Step 4 : File Names
    image_name = f"brain_{i:05}.png"
    mask_name = f"mask_{i:05}.png"

    # Step 5 : Save Image & Mask
    cv2.imwrite(
        os.path.join(IMAGE_FOLDER, image_name),
        image
    )

    cv2.imwrite(
        os.path.join(MASK_FOLDER, mask_name),
        mask
    )

    # Step 6 : Save Metadata
    metadata.append({
        "Image": image_name,
        "Mask": mask_name,
        "Tumor Type": tumor_info["tumor_type"],
        "Center X": tumor_info["center"][0],
        "Center Y": tumor_info["center"][1],
        "Radius": tumor_info["radius"]
    })

    # Progress Display
    if (i + 1) % 100 == 0:
        print(f"Generated {i + 1}/{NUMBER_OF_IMAGES} images...")

# ======================================
# Save Metadata CSV
# ======================================

df = pd.DataFrame(metadata)

df.to_csv(
    "dataset/metadata.csv",
    index=False
)

print("=" * 60)
print(" Dataset Generated Successfully!")
print(f" Total Images : {NUMBER_OF_IMAGES}")
print(f" Images Folder : {IMAGE_FOLDER}")
print(f" Masks Folder  : {MASK_FOLDER}")
print(" Metadata Saved : dataset/metadata.csv")
print("=" * 60)