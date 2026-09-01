import os
import cv2
import numpy as np
import tensorflow as tf

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

IMAGE_DIR = os.path.join(BASE_DIR, "dataset", "images")
MASK_DIR = os.path.join(BASE_DIR, "dataset", "masks")
MODEL_PATH = os.path.join(BASE_DIR, "models", "segmentation", "unet.keras")

# CHANGED: Match the 128x128 shape your model was trained on
IMG_SIZE = 128  


# ============================================================
# DICE COEFFICIENT
# ============================================================

def dice_coefficient(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    intersection = np.sum(y_true * y_pred)

    return (2.0 * intersection + 1e-7) / (
        np.sum(y_true) + np.sum(y_pred) + 1e-7
    )


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        return None

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image.astype(np.float32) / 255.0

    return image.reshape(IMG_SIZE, IMG_SIZE, 1)


# ============================================================
# LOAD MASK
# ============================================================

def load_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        return None

    mask = cv2.resize(
        mask,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_NEAREST
    )

    mask = (mask > 127).astype(np.float32)

    return mask.reshape(IMG_SIZE, IMG_SIZE, 1)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 50)
    print("U-NET EVALUATION")
    print("=" * 50)

    # Check model
    if not os.path.exists(MODEL_PATH):
        print("ERROR: U-Net model not found:")
        print(MODEL_PATH)
        return

    # Load model
    print("\nLoading U-Net model...")
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print("Model loaded successfully.")

    # Get images
    image_files = [
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
        and f.startswith("brain_")
    ]

    image_files.sort()

    print(f"\nImages found: {len(image_files)}")

    # ========================================================
    # CREATE IMAGE-MASK PAIRS
    # brain_00003.png -> mask_00003.png
    # ========================================================

    pairs = []

    for image_file in image_files:

        number = image_file.replace("brain_", "")
        mask_file = "mask_" + number

        image_path = os.path.join(IMAGE_DIR, image_file)
        mask_path = os.path.join(MASK_DIR, mask_file)

        if os.path.exists(mask_path):
            pairs.append((image_path, mask_path))

    print(f"Masks found: {len(os.listdir(MASK_DIR))}")
    print(f"Matching pairs: {len(pairs)}")

    if len(pairs) == 0:
        print("\nERROR: No matching image-mask pairs found.")
        print("Expected format:")
        print("  brain_00000.png")
        print("  mask_00000.png")
        return

    # ========================================================
    # EVALUATE
    # ========================================================

    dice_scores = []

    print("\nEvaluating...")
    print("-" * 50)

    for i, (image_path, mask_path) in enumerate(pairs):

        image = load_image(image_path)
        mask = load_mask(mask_path)

        if image is None or mask is None:
            continue

        # Prediction
        prediction = model.predict(
            np.expand_dims(image, axis=0),
            verbose=0
        )[0]

        # Convert probability to binary mask
        prediction = (prediction > 0.5).astype(np.float32)

        # Dice
        dice = dice_coefficient(mask, prediction)
        dice_scores.append(dice)

        if (i + 1) % 10 == 0:
            print(
                f"Processed {i + 1}/{len(pairs)} "
                f"| Dice: {dice:.4f}"
            )

    # ========================================================
    # RESULTS
    # ========================================================

    if len(dice_scores) == 0:
        print("\nNo images could be evaluated.")
        return

    mean_dice = np.mean(dice_scores)

    print("\n" + "=" * 50)
    print("U-NET EVALUATION RESULTS")
    print("=" * 50)

    print(f"Images evaluated : {len(dice_scores)}")
    print(f"Mean Dice Score  : {mean_dice:.4f}")
    print(f"Dice Percentage  : {mean_dice * 100:.2f}%")

    print("=" * 50)
    print("EVALUATION COMPLETED")
    print("=" * 50)


if __name__ == "__main__":
    main()