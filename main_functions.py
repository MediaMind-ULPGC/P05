from transformations.transformation_factory import TransformationsFactory
import random
import cv2 as cv
import numpy as np

""" Aplicar transformaciones aleatorias a una imagen """
def transform_img(img):

    img1 = TransformationsFactory.initialize_transformation('Traslacion').apply(img)
    img2 = TransformationsFactory.initialize_transformation('Rotacion').apply(img)
    img3 = TransformationsFactory.initialize_transformation('Escalado').apply(img)

    img4, img5 = TransformationsFactory.initialize_transformation('Distorciones').apply(img)
    random_choice = random.choice([img1, img2, img3])
    img6 = TransformationsFactory.initialize_transformation('Ruido').apply(random_choice)
    
    return img1, img2, img3, img4, img5, img6


""" Reducir el ruido de una imagen """
def reduce_noise(img):
    for i in range(10):
        img = cv.medianBlur(img, 3)
    return img


""" Aplicar SIFT a una imagen """
def apply_sift(sift, img_orig, n=10):
    founded_matches = []

    img1, img2, img3, img4, img5, img6 = transform_img(img_orig)
    img6 = reduce_noise(img6)
    interest_area = cv.imread('interest_area.png', cv.IMREAD_GRAYSCALE)
    keypoints1, descriptors1 = sift.detectAndCompute(interest_area, None)

    for img in [img1, img2, img3,  img4, img5, img6]:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        try:
            keypoints2, descriptors2 = sift.detectAndCompute(img, None)
            bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)

            matches = bf.match(descriptors1, descriptors2)
            matches = sorted(matches, key=lambda x: x.distance)
            best_matches = matches[:n]
            img_matches = cv.drawMatches(interest_area, keypoints1, img, keypoints2, best_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            founded_matches.append(cv.cvtColor(img_matches, cv.COLOR_BGR2RGB))
        
        except:
            print("Error")
            founded_matches.append(img)
    
    return founded_matches


""" Alinear una imagen """
def align_image(img_orig, nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma, n=10):

    img_orig = cv.cvtColor(img_orig, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create(nfeatures=nfeatures, nOctaveLayers=nOctaveLayers, contrastThreshold=contrastThreshold, edgeThreshold=edgeThreshold, sigma=sigma)

    img1, img2, img3, _, _, _ = transform_img(img_orig)
    transformed_img = random.choice([img1, img2, img3])
    if len(transformed_img.shape) == 3:
        transformed_img = cv.cvtColor(transformed_img, cv.COLOR_BGR2GRAY)

    keyponits1, descriptors1 = sift.detectAndCompute(img_orig, None)
    keyponits2, descriptors2 = sift.detectAndCompute(transformed_img, None)

    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)

    try:
        matches = bf.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)
        best_matches = matches[:n]
        src_pts = np.float32([keyponits1[m.queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keyponits2[m.trainIdx].pt for m in best_matches]).reshape(-1, 1, 2)

        M, mask = cv.findHomography(dst_pts, src_pts, cv.RANSAC, 5.0)

        height, width = img_orig.shape[:2]
        size = (width, height)
        aligned_img = cv.warpPerspective(transformed_img, M, size)

        aligned_img = cv.cvtColor(aligned_img, cv.COLOR_GRAY2BGR)
        transformed_img = cv.cvtColor(transformed_img, cv.COLOR_GRAY2BGR)

        return aligned_img, transformed_img
        
    except:
        return None, None
