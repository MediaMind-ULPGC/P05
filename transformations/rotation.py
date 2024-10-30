import random
import cv2 as cv

class Rotation:

    @staticmethod
    def apply(img):
        height, width = img.shape[:2]
        center = (width // 2, height // 2)
        size = (width, height)
        R = cv.getRotationMatrix2D(center, random.randint(1,360), 1)
        return cv.warpAffine(img, R, size)