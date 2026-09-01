import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "classifier",
    "tumor_classifier.keras"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "reports",
    "explainability"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# CLASSES
# ============================================================

CLASS_NAMES = [
    "No Tumor",
    "Glioma",
    "Meningioma",
    "Pituitary"
]


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("GRAD-CAM EXPLAINABLE AI")
print("=" * 60)

print("\nLoading classifier model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully.")


# ============================================================
# FIND LAST CONVOLUTIONAL LAYER
# ============================================================

last_conv_layer = None

for layer in reversed(model.layers):

    if isinstance(layer, tf.keras.layers.Conv2D):

        last_conv_layer = layer
        break


if last_conv_layer is None:

    raise ValueError(
        "No Conv2D layer found."
    )


print(
    "Grad-CAM layer:",
    last_conv_layer.name
)


# ============================================================
# GRAD-CAM
# ============================================================

def make_gradcam_heatmap(image_array):

    with tf.GradientTape() as tape:

        # ----------------------------------------------------
        # Manually run through the network
        # ----------------------------------------------------

        x = image_array

        conv_output = None

        for layer in model.layers:

            x = layer(x)

            if layer == last_conv_layer:

                conv_output = x

        predictions = x

        # ----------------------------------------------------
        # Predicted class
        # ----------------------------------------------------

        predicted_class = tf.argmax(
            predictions[0]
        )

        class_score = predictions[
            0,
            predicted_class
        ]

    # --------------------------------------------------------
    # Gradients
    # --------------------------------------------------------

    gradients = tape.gradient(
        class_score,
        conv_output
    )

    if gradients is None:

        raise RuntimeError(
            "Gradients are None. "
            "Grad-CAM could not calculate gradients."
        )

    # --------------------------------------------------------
    # Global average pooling
    # --------------------------------------------------------

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    # Remove batch dimension
    conv_output = conv_output[0]

    # --------------------------------------------------------
    # Weighted feature maps
    # --------------------------------------------------------

    heatmap = tf.reduce_sum(
        conv_output *
        pooled_gradients,
        axis=-1
    )

    # ReLU
    heatmap = tf.maximum(
        heatmap,
        0
    )

    # Normalize
    maximum = tf.reduce_max(
        heatmap
    )

    if maximum > 0:

        heatmap = (
            heatmap /
            maximum
        )

    return (
        heatmap.numpy(),
        int(predicted_class.numpy())
    )


# ============================================================
# MAIN
# ============================================================

def generate_gradcam(image_path):

    print("\nLoading image:")
    print(image_path)

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    original = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if original is None:

        raise FileNotFoundError(
            f"Could not read image:\n{image_path}"
        )

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    image = cv2.resize(
        original,
        (128, 128),
        interpolation=cv2.INTER_AREA
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    image = (
        image.astype(
            np.float32
        ) / 255.0
    )

    # --------------------------------------------------------
    # Shape: (1,128,128,1)
    # --------------------------------------------------------

    image = np.expand_dims(
        image,
        axis=-1
    )

    image_array = np.expand_dims(
        image,
        axis=0
    )

    print(
        "Model input shape:",
        image_array.shape
    )

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    predictions = model.predict(
        image_array,
        verbose=0
    )

    predicted_class = np.argmax(
        predictions[0]
    )

    confidence = (
        predictions[
            0,
            predicted_class
        ] * 100
    )

    predicted_label = CLASS_NAMES[
        predicted_class
    ]

    print("\n" + "=" * 60)
    print("CLASSIFICATION")
    print("=" * 60)

    print(
        f"Prediction : {predicted_label}"
    )

    print(
        f"Confidence : {confidence:.2f}%"
    )

    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    heatmap, _ = make_gradcam_heatmap(
        tf.convert_to_tensor(
            image_array,
            dtype=tf.float32
        )
    )

    # --------------------------------------------------------
    # Resize heatmap
    # --------------------------------------------------------

    heatmap = cv2.resize(
        heatmap,
        (
            original.shape[1],
            original.shape[0]
        ),
        interpolation=cv2.INTER_LINEAR
    )

    # --------------------------------------------------------
    # Convert to 0-255
    # --------------------------------------------------------

    heatmap_uint8 = np.uint8(
        255 * heatmap
    )

    # --------------------------------------------------------
    # Color heatmap
    # --------------------------------------------------------

    colored_heatmap = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    # --------------------------------------------------------
    # Original MRI → BGR
    # --------------------------------------------------------

    original_color = cv2.cvtColor(
        original,
        cv2.COLOR_GRAY2BGR
    )

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    overlay = cv2.addWeighted(
        original_color,
        0.60,
        colored_heatmap,
        0.40,
        0
    )

    # ========================================================
    # OUTPUT FILES
    # ========================================================

    image_name = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    original_path = os.path.join(
        OUTPUT_DIR,
        f"{image_name}_original.png"
    )

    heatmap_path = os.path.join(
        OUTPUT_DIR,
        f"{image_name}_gradcam_heatmap.png"
    )

    overlay_path = os.path.join(
        OUTPUT_DIR,
        f"{image_name}_gradcam.png"
    )

    result_path = os.path.join(
        OUTPUT_DIR,
        f"{image_name}_gradcam_result.png"
    )

    # --------------------------------------------------------
    # Save images
    # --------------------------------------------------------

    cv2.imwrite(
        original_path,
        original
    )

    cv2.imwrite(
        heatmap_path,
        colored_heatmap
    )

    cv2.imwrite(
        overlay_path,
        overlay
    )

    # ========================================================
    # FINAL FIGURE
    # ========================================================

    plt.figure(
        figsize=(15, 5)
    )

    # Original
    plt.subplot(
        1,
        3,
        1
    )

    plt.imshow(
        original,
        cmap="gray"
    )

    plt.title(
        "Original MRI"
    )

    plt.axis("off")

    # Heatmap
    plt.subplot(
        1,
        3,
        2
    )

    plt.imshow(
        heatmap,
        cmap="jet"
    )

    plt.title(
        "Grad-CAM Heatmap"
    )

    plt.axis("off")

    # Overlay
    plt.subplot(
        1,
        3,
        3
    )

    plt.imshow(
        cv2.cvtColor(
            overlay,
            cv2.COLOR_BGR2RGB
        )
    )

    plt.title(
        f"{predicted_label} "
        f"({confidence:.2f}%)"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        result_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print("\n" + "=" * 60)
    print("GRAD-CAM COMPLETED")
    print("=" * 60)

    print(
        f"Prediction : {predicted_label}"
    )

    print(
        f"Confidence : {confidence:.2f}%"
    )

    print("\nSaved files:")

    print(
        f"Original : {original_path}"
    )

    print(
        f"Heatmap  : {heatmap_path}"
    )

    print(
        f"Overlay  : {overlay_path}"
    )

    print(
        f"Result   : {result_path}"
    )

    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "\nUsage:"
        )

        print(
            "python models/classifier/gradcam.py "
            "dataset/images/brain_00003.png"
        )

        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.isabs(image_path):

        image_path = os.path.join(
            PROJECT_ROOT,
            image_path
        )

    generate_gradcam(
        image_path
    )