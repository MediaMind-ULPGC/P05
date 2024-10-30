import cv2 as cv
import random

class Scalated:

    @staticmethod
    def apply(img):
        height, width = img.shape[:2]
        size = (width, height)
        center = (width // 2, height // 2)

        scale_factor = random.randint(1, width/2) / 50.0  
        S = cv.getRotationMatrix2D(center, 0, scale_factor)
        return cv.warpAffine(img, S, size)