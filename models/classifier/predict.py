import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

# ======================================
# Add project root
# ======================================

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )
    )
)

# ======================================
# Load Trained Model
# ======================================

model = load_model(
    "models/classifier/tumor_classifier.keras"
)

# ======================================
# Class Labels
# ======================================

labels = [
    "No Tumor",
    "Glioma",
    "Meningioma",
    "Pituitary"
]

# ======================================
# Image Path
# ======================================

image_path = input("Enter MRI image path: ")

# ======================================
# Read Image
# ======================================

image = cv2.imread(
    image_path,
    cv2.IMREAD_GRAYSCALE
)

if image is None:
    print("Image not found!")
    exit()

original = image.copy()

# ======================================
# Preprocess
# ======================================

image = cv2.resize(
    image,
    (128, 128)
)

image = image.astype(np.float32) / 255.0

image = np.expand_dims(image, axis=-1)

image = np.expand_dims(image, axis=0)

# ======================================
# Prediction
# ======================================

prediction = model.predict(image, verbose=0)

predicted_class = np.argmax(prediction)

confidence = np.max(prediction) * 100

tumor = labels[predicted_class]

# ======================================
# Output
# ======================================

print("=" * 50)
print("Prediction :", tumor)
print(f"Confidence : {confidence:.2f}%")
print("=" * 50)

# ======================================
# Show Image
# ======================================

plt.imshow(original, cmap="gray")

plt.title(
    f"{tumor} ({confidence:.2f}%)"
)

plt.axis("off")

plt.show()