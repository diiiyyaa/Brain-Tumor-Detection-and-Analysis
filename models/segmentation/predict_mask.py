import os
import sys
import cv2
import numpy as np
import tensorflow as tf


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "segmentation",
    "unet.keras"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "reports",
    "segmentation"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading U-Net model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully.")


# ============================================================
# PREDICT MASK
# ============================================================

def predict_mask(image_path):

    print(f"\nLoading image: {image_path}")

    # Read MRI as grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    original = image.copy()

    # Get model input size
    input_shape = model.input_shape

    height = input_shape[1]
    width = input_shape[2]

    print(f"Model input size: {width} x {height}")

    # Resize
    image_resized = cv2.resize(
        image,
        (width, height),
        interpolation=cv2.INTER_AREA
    )

    # Normalize
    image_normalized = image_resized.astype(
        np.float32
    ) / 255.0

    # Add channel dimension
    image_input = np.expand_dims(
        image_normalized,
        axis=-1
    )

    # Add batch dimension
    image_input = np.expand_dims(
        image_input,
        axis=0
    )

    # ========================================================
    # PREDICTION
    # ========================================================

    prediction = model.predict(
        image_input,
        verbose=0
    )

    # Remove batch/channel dimensions
    prediction = prediction[0]

    if prediction.ndim == 3:
        prediction = prediction[:, :, 0]

    # Convert probability map to binary mask
    mask = (prediction > 0.5).astype(np.uint8) * 255

    # Resize mask back to original image size
    mask = cv2.resize(
        mask,
        (original.shape[1], original.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    # ========================================================
    # CREATE OVERLAY
    # ========================================================

    original_color = cv2.cvtColor(
        original,
        cv2.COLOR_GRAY2BGR
    )

    # Create colored tumor overlay
    overlay = original_color.copy()

    tumor_pixels = mask > 0

    # Red tumor region
    overlay[tumor_pixels] = [0, 0, 255]

    # Blend original + tumor overlay
    blended = cv2.addWeighted(
        original_color,
        0.65,
        overlay,
        0.35,
        0
    )

    # ========================================================
    # CREATE SIDE-BY-SIDE RESULT
    # ========================================================

    original_display = cv2.cvtColor(
        original,
        cv2.COLOR_GRAY2BGR
    )

    mask_display = cv2.cvtColor(
        mask,
        cv2.COLOR_GRAY2BGR
    )

    result = np.hstack([
        original_display,
        mask_display,
        blended
    ])

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    base_name = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    original_path = os.path.join(
        OUTPUT_DIR,
        f"{base_name}_original.png"
    )

    mask_path = os.path.join(
        OUTPUT_DIR,
        f"{base_name}_predicted_mask.png"
    )

    overlay_path = os.path.join(
        OUTPUT_DIR,
        f"{base_name}_overlay.png"
    )

    result_path = os.path.join(
        OUTPUT_DIR,
        f"{base_name}_segmentation_result.png"
    )

    cv2.imwrite(original_path, original)
    cv2.imwrite(mask_path, mask)
    cv2.imwrite(overlay_path, blended)
    cv2.imwrite(result_path, result)

    # ========================================================
    # RESULTS
    # ========================================================

    tumor_pixels_count = np.sum(mask > 0)
    total_pixels = mask.shape[0] * mask.shape[1]

    tumor_percentage = (
        tumor_pixels_count / total_pixels
    ) * 100

    print("\n" + "=" * 50)
    print("U-NET PREDICTION COMPLETED")
    print("=" * 50)

    print(f"Image                 : {base_name}")
    print(f"Tumor pixels          : {tumor_pixels_count}")
    print(f"Tumor area percentage : {tumor_percentage:.2f}%")

    print("\nSaved files:")

    print(f"Original : {original_path}")
    print(f"Mask     : {mask_path}")
    print(f"Overlay  : {overlay_path}")
    print(f"Result   : {result_path}")

    print("=" * 50)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) != 2:

        print("\nUsage:")
        print(
            "python models/segmentation/predict_mask.py "
            "path/to/image.png"
        )

        print("\nExample:")
        print(
            "python models/segmentation/predict_mask.py "
            "dataset/images/brain_0003.png"
        )

        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.isabs(image_path):
        image_path = os.path.join(
            PROJECT_ROOT,
            image_path
        )

    predict_mask(image_path)