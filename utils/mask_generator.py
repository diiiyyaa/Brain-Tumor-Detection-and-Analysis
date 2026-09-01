import cv2
import numpy as np
import random

class MaskGenerator:

    def create_mask(
        self,
        image_size,
        tumor_type,
        center,
        radius
    ):

        mask = np.zeros(
            (image_size, image_size),
            dtype=np.uint8
        )

        # Healthy Brain
        if tumor_type == "No Tumor":
            return mask

        x, y = center

        # -------------------------
        # Glioma (Irregular)
        # -------------------------

        if tumor_type == "Glioma":

            points = []

            for angle in range(0, 360, 20):

                r = radius + random.randint(-6, 6)

                px = int(
                    x + r * np.cos(np.radians(angle))
                )

                py = int(
                    y + r * np.sin(np.radians(angle))
                )

                points.append([px, py])

            points = np.array(points, np.int32)

            cv2.fillPoly(
                mask,
                [points],
                255
            )

        # -------------------------
        # Meningioma (Ellipse)
        # -------------------------

        elif tumor_type == "Meningioma":

            axes = (
                radius,
                random.randint(
                    radius - 3,
                    radius + 6
                )
            )

            angle = random.randint(0,180)

            cv2.ellipse(
                mask,
                (x,y),
                axes,
                angle,
                0,
                360,
                255,
                -1
            )

        # -------------------------
        # Pituitary
        # -------------------------

        else:

            cv2.circle(
                mask,
                (x,y),
                radius,
                255,
                -1
            )

        return mask