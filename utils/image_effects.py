import cv2
import numpy as np


class ImageEffects:

    def __init__(self):
        pass

    def add_gaussian_noise(self, image):

        noise = np.random.normal(
            0,
            8,
            image.shape
        )

        noisy = image.astype(np.float32)

        noisy += noise

        noisy = np.clip(
            noisy,
            0,
            255
        )

        return noisy.astype(np.uint8)

    def gaussian_blur(self, image):

        return cv2.GaussianBlur(
            image,
            (5,5),
            0
        )

    def clahe(self, image):

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8,8)
        )

        return clahe.apply(image)

    def normalize(self, image):

        return cv2.normalize(

            image,

            None,

            0,

            255,

            cv2.NORM_MINMAX

        )