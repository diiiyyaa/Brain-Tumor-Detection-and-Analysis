import os
import cv2
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


class DatasetLoader:

    def __init__(
        self,
        image_folder="dataset/images",
        metadata_file="dataset/metadata.csv",
        image_size=128
    ):

        self.image_folder = image_folder
        self.metadata_file = metadata_file
        self.image_size = image_size

        self.label_map = {
            "No Tumor": 0,
            "Glioma": 1,
            "Meningioma": 2,
            "Pituitary": 3
        }

    def load_dataset(self):

        df = pd.read_csv(self.metadata_file)

        images = []
        labels = []

        for _, row in df.iterrows():

            image_path = os.path.join(
                self.image_folder,
                row["Image"]
            )

            image = cv2.imread(
                image_path,
                cv2.IMREAD_GRAYSCALE
            )

            if image is None:
                continue

            image = cv2.resize(
                image,
                (self.image_size, self.image_size)
            )

            image = image.astype(np.float32) / 255.0

            image = np.expand_dims(
                image,
                axis=-1
            )

            label = self.label_map[
                row["Tumor Type"]
            ]

            images.append(image)
            labels.append(label)

        X = np.array(images)

        y = to_categorical(
            labels,
            num_classes=4
        )

        return train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=labels
        )