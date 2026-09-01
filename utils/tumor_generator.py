import cv2
import numpy as np
import random

class TumorGenerator:

    def __init__(self):

        # Balanced 4-class dataset
        self.tumor_types = [
            "No Tumor",
            "Glioma",
            "Meningioma",
            "Pituitary"
        ]

    def random_type(self):
        return random.choice(self.tumor_types)

    def draw(self, image):

        tumor = self.random_type()

        # Healthy brain
        if tumor == "No Tumor":
            return image, {
                "tumor_type": "No Tumor",
                "center": (-1, -1),
                "radius": 0
            }

        x = random.randint(75, 180)
        y = random.randint(70, 185)

        if tumor == "Glioma":
            radius = random.randint(22, 35)
            color = random.randint(190, 220)

            pts = []
            for angle in range(0, 360, 20):
                r = radius + random.randint(-7, 7)
                px = int(x + r*np.cos(np.radians(angle)))
                py = int(y + r*np.sin(np.radians(angle)))
                pts.append([px, py])

            pts = np.array(pts, np.int32)
            cv2.fillPoly(image, [pts], color)

        elif tumor == "Meningioma":

            radius = random.randint(15, 25)
            color = random.randint(215, 235)

            axes = (
                radius,
                random.randint(radius-3, radius+6)
            )

            angle = random.randint(0,180)

            cv2.ellipse(
                image,
                (x,y),
                axes,
                angle,
                0,
                360,
                color,
                -1
            )

        else:

            radius = random.randint(8,15)
            color = random.randint(235,250)

            x = random.randint(105,150)
            y = random.randint(160,195)

            cv2.circle(
                image,
                (x,y),
                radius,
                color,
                -1
            )

        return image,{
            "tumor_type":tumor,
            "center":(x,y),
            "radius":radius
        }