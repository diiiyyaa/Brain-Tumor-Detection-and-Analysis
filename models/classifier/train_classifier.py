import os
import sys

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

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)

loader = DatasetLoader()

X_train, X_test, y_train, y_test = loader.load_dataset()

model = Sequential()

model.add(
    Conv2D(
        32,
        (3,3),
        activation="relu",
        input_shape=(128,128,1)
    )
)

model.add(
    MaxPooling2D()
)

model.add(
    Conv2D(
        64,
        (3,3),
        activation="relu"
    )
)

model.add(
    MaxPooling2D()
)

model.add(
    Conv2D(
        128,
        (3,3),
        activation="relu"
    )
)

model.add(
    MaxPooling2D()
)

model.add(
    Flatten()
)

model.add(
    Dense(
        256,
        activation="relu"
    )
)

model.add(
    Dropout(0.5)
)

model.add(
    Dense(
        4,
        activation="softmax"
    )
)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

checkpoint = ModelCheckpoint(
    "models/classifier/tumor_classifier.keras",
    save_best_only=True,
    monitor="val_accuracy"
)

early = EarlyStopping(
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=32,
    callbacks=[
        checkpoint,
        early
    ]
)

loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print("="*40)
print("Test Accuracy :", accuracy)
print("="*40)