import os
import cv2
import numpy as np


class SegmentationLoader:

    def __init__(
        self,
        image_folder="dataset/images",
        mask_folder="dataset/masks",
        image_size=256
    ):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.image_size = image_size

    def load_data(self, max_samples=None):

        images = []
        masks = []

        image_files = sorted(
            [
                file
                for file in os.listdir(self.image_folder)
                if file.lower().endswith(".png")
            ]
        )

        # For testing, load only a few images
        if max_samples is not None:
            image_files = image_files[:max_samples]

        print("=" * 50)
        print("Loading Segmentation Dataset")
        print("Images to load:", len(image_files))
        print("=" * 50)

        for image_file in image_files:

            image_path = os.path.join(
                self.image_folder,
                image_file
            )

            mask_file = image_file.replace(
                "brain_",
                "mask_"
            )

            mask_path = os.path.join(
                self.mask_folder,
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

            if image is None:
                print("Could not read:", image_file)
                continue

            if mask is None:
                print("Could not read:", mask_file)
                continue

            image = cv2.resize(
                image,
                (self.image_size, self.image_size)
            )

            mask = cv2.resize(
                mask,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_NEAREST
            )

            image = image.astype(
                np.float32
            ) / 255.0

            mask = mask.astype(
                np.float32
            ) / 255.0

            mask = (
                mask > 0.5
            ).astype(np.float32)

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

        images = np.array(
            images,
            dtype=np.float32
        )

        masks = np.array(
            masks,
            dtype=np.float32
        )

        print("=" * 50)
        print("Dataset Loaded Successfully")
        print("Images:", images.shape)
        print("Masks :", masks.shape)
        print("=" * 50)

        return images, masks