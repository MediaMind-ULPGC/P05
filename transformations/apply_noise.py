import numpy as np

class ApplyNoise:

    @staticmethod
    def apply(img):
        img = img + np.random.normal(0, 50, img.shape)
        noisy = np.clip(img, 0, 255)

        return noisy.astype(np.uint8)