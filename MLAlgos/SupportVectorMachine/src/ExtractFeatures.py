import cv2
import numpy as np
from skimage.feature import hog

def GetGradientFromImg(img_list):
    mag_list = []
    angle_list = []
    for img in img_list:
        # Calculate gradient
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)

        # Calculate gradient magnitude and direction ( in degrees )
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

        mag_list.append(mag)
        angle_list.append(angle)

    return mag_list, angle_list


def GetHogFeatures(img_list):
    fd_list = []
    hog_img_list = []
    for img in img_list:
        fd, hog_img = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                          visualize=True, multichannel=True)
        fd_list.append(fd)
        hog_img_list.append(hog_img)

    return fd_list, hog_img_list
