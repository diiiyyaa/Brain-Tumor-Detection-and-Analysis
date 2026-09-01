import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from tensorflow.keras.models import load_model

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )
    )
)

from utils.dataset_loader import DatasetLoader


# ==========================
# Load Dataset
# ==========================

loader = DatasetLoader()

X_train, X_test, y_train, y_test = loader.load_dataset()

# ==========================
# Load Model
# ==========================

model = load_model(
    "models/classifier/tumor_classifier.keras"
)

# ==========================
# Prediction
# ==========================

predictions = model.predict(X_test)

y_pred = np.argmax(predictions, axis=1)

y_true = np.argmax(y_test, axis=1)

# ==========================
# Confusion Matrix
# ==========================

labels = [
    "No Tumor",
    "Glioma",
    "Meningioma",
    "Pituitary"
]

cm = confusion_matrix(
    y_true,
    y_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

fig, ax = plt.subplots(figsize=(8,8))

disp.plot(
    cmap="Blues",
    ax=ax,
    colorbar=False
)

plt.title("Brain Tumor Confusion Matrix")

plt.savefig(
    "models/classifier/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ==========================
# Classification Report
# ==========================

print("="*60)

print("Classification Report")

print("="*60)

print(

    classification_report(
        y_true,
        y_pred,
        target_names=labels
    )

)