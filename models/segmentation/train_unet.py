import os
import cv2
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

IMAGE_FOLDER = os.path.join(
    PROJECT_ROOT, "dataset", "images"
)

MASK_FOLDER = os.path.join(
    PROJECT_ROOT, "dataset", "masks"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "segmentation",
    "unet.keras"
)

# ============================================================
# FAST SETTINGS
# ============================================================

IMAGE_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 3

# Only use this many images for FAST training
MAX_SAMPLES = 1000

RANDOM_STATE = 42

# ============================================================
# DICE COEFFICIENT
# ============================================================

def dice_coefficient(y_true, y_pred):

    smooth = 1e-6

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(
        y_true * y_pred
    )

    return (
        (2.0 * intersection + smooth)
        /
        (
            tf.reduce_sum(y_true)
            + tf.reduce_sum(y_pred)
            + smooth
        )
    )


# ============================================================
# DICE LOSS
# ============================================================

def dice_loss(y_true, y_pred):

    return 1.0 - dice_coefficient(
        y_true,
        y_pred
    )


# ============================================================
# SMALL CONVOLUTION BLOCK
# ============================================================

def conv_block(x, filters):

    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        activation="relu"
    )(x)

    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        activation="relu"
    )(x)

    return x


# ============================================================
# FAST U-NET
# ============================================================

def build_unet():

    inputs = layers.Input(
        shape=(
            IMAGE_SIZE,
            IMAGE_SIZE,
            1
        )
    )

    # ---------------- Encoder ----------------

    c1 = conv_block(inputs, 8)
    p1 = layers.MaxPooling2D(2)(c1)

    c2 = conv_block(p1, 16)
    p2 = layers.MaxPooling2D(2)(c2)

    c3 = conv_block(p2, 32)
    p3 = layers.MaxPooling2D(2)(c3)

    # ---------------- Bottleneck ----------------

    c4 = conv_block(p3, 64)

    # ---------------- Decoder ----------------

    u5 = layers.Conv2DTranspose(
        32,
        2,
        strides=2,
        padding="same"
    )(c4)

    u5 = layers.concatenate(
        [u5, c3]
    )

    c5 = conv_block(u5, 32)

    u6 = layers.Conv2DTranspose(
        16,
        2,
        strides=2,
        padding="same"
    )(c5)

    u6 = layers.concatenate(
        [u6, c2]
    )

    c6 = conv_block(u6, 16)

    u7 = layers.Conv2DTranspose(
        8,
        2,
        strides=2,
        padding="same"
    )(c6)

    u7 = layers.concatenate(
        [u7, c1]
    )

    c7 = conv_block(u7, 8)

    outputs = layers.Conv2D(
        1,
        1,
        activation="sigmoid"
    )(c7)

    return Model(
        inputs,
        outputs,
        name="Fast_Brain_Tumor_U_Net"
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_data(image_files):

    images = []
    masks = []

    print("\nLoading images and masks...")
    print("-" * 50)

    for i, image_file in enumerate(image_files):

        image_path = os.path.join(
            IMAGE_FOLDER,
            image_file
        )

        mask_file = image_file.replace(
            "brain_",
            "mask_"
        )

        mask_path = os.path.join(
            MASK_FOLDER,
            mask_file
        )

        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        mask = cv2.imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None or mask is None:
            continue

        # Resize image
        image = cv2.resize(
            image,
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            interpolation=cv2.INTER_AREA
        )

        # Resize mask
        mask = cv2.resize(
            mask,
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            interpolation=cv2.INTER_NEAREST
        )

        # Normalize image
        image = image.astype(
            np.float32
        ) / 255.0

        # Binary mask
        mask = (
            mask > 127
        ).astype(
            np.float32
        )

        # Add channel
        image = np.expand_dims(
            image,
            axis=-1
        )

        mask = np.expand_dims(
            mask,
            axis=-1
        )

        images.append(image)
        masks.append(mask)

        if (i + 1) % 100 == 0:
            print(
                f"Loaded {i + 1}/{len(image_files)}"
            )

    images = np.array(
        images,
        dtype=np.float32
    )

    masks = np.array(
        masks,
        dtype=np.float32
    )

    print(
        f"\nImages loaded: {len(images)}"
    )

    print(
        f"Image shape: {images.shape}"
    )

    print(
        f"Mask shape: {masks.shape}"
    )

    return images, masks


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("FAST BRAIN TUMOR U-NET SEGMENTATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Check folders
    # --------------------------------------------------------

    if not os.path.exists(IMAGE_FOLDER):

        raise FileNotFoundError(
            f"Image folder not found:\n{IMAGE_FOLDER}"
        )

    if not os.path.exists(MASK_FOLDER):

        raise FileNotFoundError(
            f"Mask folder not found:\n{MASK_FOLDER}"
        )

    # --------------------------------------------------------
    # Find images
    # --------------------------------------------------------

    image_files = [
        f
        for f in os.listdir(IMAGE_FOLDER)
        if f.lower().endswith(".png")
        and f.startswith("brain_")
    ]

    image_files.sort()

    print(
        f"\nTotal images found: {len(image_files)}"
    )

    # --------------------------------------------------------
    # Check masks
    # --------------------------------------------------------

    valid_files = []

    for image_file in image_files:

        mask_file = image_file.replace(
            "brain_",
            "mask_"
        )

        mask_path = os.path.join(
            MASK_FOLDER,
            mask_file
        )

        if os.path.exists(mask_path):

            valid_files.append(
                image_file
            )

    print(
        f"Valid image-mask pairs: {len(valid_files)}"
    )

    # --------------------------------------------------------
    # LIMIT DATASET FOR FAST TRAINING
    # --------------------------------------------------------

    if len(valid_files) > MAX_SAMPLES:

        np.random.seed(
            RANDOM_STATE
        )

        valid_files = np.random.choice(
            valid_files,
            MAX_SAMPLES,
            replace=False
        ).tolist()

    print(
        f"Using samples for training: {len(valid_files)}"
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    X, Y = load_data(
        valid_files
    )

    if len(X) == 0:

        raise RuntimeError(
            "No valid images were loaded."
        )

    # --------------------------------------------------------
    # Train / validation / test split
    # --------------------------------------------------------

    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X,
        Y,
        test_size=0.20,
        random_state=RANDOM_STATE
    )

    X_val, X_test, Y_val, Y_test = train_test_split(
        X_temp,
        Y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE
    )

    print("\nDataset Split")
    print("-" * 50)

    print(
        "Training   :",
        len(X_train)
    )

    print(
        "Validation :",
        len(X_val)
    )

    print(
        "Testing    :",
        len(X_test)
    )

    # --------------------------------------------------------
    # TensorFlow datasets
    # --------------------------------------------------------

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (X_train, Y_train)
    )

    train_dataset = train_dataset.shuffle(
        buffer_size=len(X_train)
    )

    train_dataset = train_dataset.batch(
        BATCH_SIZE
    )

    train_dataset = train_dataset.prefetch(
        tf.data.AUTOTUNE
    )

    val_dataset = tf.data.Dataset.from_tensor_slices(
        (X_val, Y_val)
    )

    val_dataset = val_dataset.batch(
        BATCH_SIZE
    )

    val_dataset = val_dataset.prefetch(
        tf.data.AUTOTUNE
    )

    test_dataset = tf.data.Dataset.from_tensor_slices(
        (X_test, Y_test)
    )

    test_dataset = test_dataset.batch(
        BATCH_SIZE
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    print("\nBuilding FAST U-Net...")

    model = build_unet()

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),

        loss=dice_loss,

        metrics=[
            dice_coefficient
        ]
    )

    model.summary()

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(MODEL_PATH),
        exist_ok=True
    )

    checkpoint = ModelCheckpoint(

        MODEL_PATH,

        monitor="val_dice_coefficient",

        mode="max",

        save_best_only=True,

        verbose=1
    )

    early_stopping = EarlyStopping(

        monitor="val_dice_coefficient",

        mode="max",

        patience=2,

        restore_best_weights=True,

        verbose=1
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("STARTING FAST U-NET TRAINING")
    print("=" * 60)

    model.fit(

        train_dataset,

        validation_data=val_dataset,

        epochs=EPOCHS,

        callbacks=[
            checkpoint,
            early_stopping
        ],

        verbose=1
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("TESTING MODEL")
    print("=" * 60)

    test_loss, test_dice = model.evaluate(
        test_dataset,
        verbose=1
    )

    print(
        f"\nTest Loss : {test_loss:.4f}"
    )

    print(
        f"Test Dice : {test_dice:.4f}"
    )

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    model.save(
        MODEL_PATH
    )

    print("\n")
    print("=" * 60)
    print("U-NET TRAINING COMPLETED")
    print("=" * 60)

    print(
        "Model saved at:"
    )

    print(
        MODEL_PATH
    )

    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()