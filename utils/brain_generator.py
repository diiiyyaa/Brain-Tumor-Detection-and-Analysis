import cv2
import numpy as np
import random


class BrainGenerator:
    """
    Generates realistic synthetic brain MRI images.
    """

    def __init__(self, image_size=256):

        self.image_size = image_size

        self.center = (
            image_size // 2,
            image_size // 2
        )

    def create_blank_image(self):

        return np.zeros(
            (self.image_size, self.image_size),
            dtype=np.uint8
        )

    def draw_brain(self, image):

        # Random brain dimensions
        width = random.randint(85, 100)
        height = random.randint(105, 120)

        # Small random rotation
        angle = random.randint(-8, 8)

        # Brain intensity
        intensity = random.randint(105, 135)

        cv2.ellipse(
            image,
            self.center,
            (width, height),
            angle,
            0,
            360,
            intensity,
            -1
        )

        return image

    def draw_skull(self, image):

        width = random.randint(98, 108)
        height = random.randint(118, 128)

        angle = random.randint(-8, 8)

        thickness = random.randint(2, 4)

        skull_intensity = random.randint(170, 200)

        cv2.ellipse(
            image,
            self.center,
            (width, height),
            angle,
            0,
            360,
            skull_intensity,
            thickness
        )

        return image

    def add_texture(self, image):

        # Gaussian Noise
        noise = np.random.normal(
            0,
            random.randint(6, 12),
            image.shape
        )

        image = image.astype(np.float32)

        image += noise

        # Brightness variation
        image += random.randint(-10, 10)

        image = np.clip(
            image,
            0,
            255
        )

        return image.astype(np.uint8)

    def add_mri_effect(self, image):

        # Slight blur
        k = random.choice([3, 5])

        image = cv2.GaussianBlur(
            image,
            (k, k),
            0
        )

        # Contrast adjustment
        alpha = random.uniform(0.9, 1.15)

        beta = random.randint(-5, 5)

        image = cv2.convertScaleAbs(
            image,
            alpha=alpha,
            beta=beta
        )

        return image

    def generate(self):

        image = self.create_blank_image()

        image = self.draw_brain(image)

        image = self.draw_skull(image)

        image = self.add_texture(image)

        image = self.add_mri_effect(image)

        return image