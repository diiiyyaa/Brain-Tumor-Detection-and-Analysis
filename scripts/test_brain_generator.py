import cv2
import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.brain_generator import BrainGenerator


brain = BrainGenerator()

image = brain.generate()

cv2.imshow("Brain", image)

cv2.waitKey(0)

cv2.destroyAllWindows()