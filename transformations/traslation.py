import random
import numpy as np
import cv2 as cv

class Traslation:

    @staticmethod
    def apply(img):
        height, width = img.shape[:2]
        size = (width, height)

        tx =  random.randint(1, width) - width
        ty = random.randint(1, height) - height
        T = np.float32([[1, 0, tx], [0, 1, ty]])
        
        return cv.warpAffine(img, T, size)
        