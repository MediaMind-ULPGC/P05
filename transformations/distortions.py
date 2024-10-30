import cv2 as cv
import numpy as np
import random

class Distortions:

    @staticmethod
    def _apply_distortion(image, k1, k2):
        h, w = image.shape[:2]
        distCoeff = np.zeros((4,1), np.float64)
        distCoeff[0,0] = k1
        distCoeff[1,0] = k2

        cam = np.eye(3, dtype=np.float32)
        cam[0,2] = w/2.0 
        cam[1,2] = h/2.0
        cam[0,0] = 10.
        cam[1,1] = 10.

        distorted_img = cv.undistort(image, cam, distCoeff)

        return distorted_img
    
    @staticmethod
    def apply(img):

        k1 = random.randint(1, 30) /100000.0
        k2 = random.randint(1, 30) /100000.0

        img_barril = Distortions._apply_distortion(img, -k1, -k2) 
        img_cojin = Distortions._apply_distortion(img, k1, k2)

        return img_barril, img_cojin

    