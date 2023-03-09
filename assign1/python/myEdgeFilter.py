# Author : Daniel Carstensen
# Date : 01/19/2023
# File name : myEdgeFilter.py
# Class : COSC83
# Purpose : Create function to detect edges in input image through blurring and sobel filtering

import numpy as np
import scipy.signal
import cv2
from myImageFilter import myImageFilter


def myEdgeFilter(img0, sigma):
    # create gaussian kernel
    gaussian = scipy.signal.gaussian(M=2*np.ceil(2*sigma)+1, std=sigma)
    gaussian = np.outer(gaussian, gaussian)
    gaussian = gaussian/gaussian.sum()

    # create sobel kernels in x and y direction
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # apply gaussian and sobel filters
    img0 = myImageFilter(img0, gaussian)
    img_x = myImageFilter(img0, sobel_x)
    img_y = myImageFilter(img0, sobel_y)

    # combine edges in x and y direction and find gradient direction
    G = np.hypot(img_x, img_y)
    theta = np.multiply(np.arctan2(img_y, img_x), 180/np.pi)
    theta[theta<0] += 180

    # dilate image corresponding to four different gradient directions
    diag_kernel = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], np.uint8)
    inv_diag_kernel = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.uint8)
    horz_kernel = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], np.uint8)
    vert_kernel = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], np.uint8)

    diag_G = cv2.dilate(G, diag_kernel)
    inv_diag_G = cv2.dilate(G, inv_diag_kernel)
    horz_G = cv2.dilate(G, horz_kernel)
    vert_G = cv2.dilate(G, vert_kernel)

    cond_list = [((theta <= 22.5) | (theta > 157.5)) & (G == horz_G),
                 (theta <= 67.5) & (theta > 22.5) & (G == inv_diag_G),
                 (theta <= 112.5) & (theta > 67.5) & (G == vert_G),
                 (theta <= 157.5) & (theta > 112.5) & (G == diag_G)]

    choice_list = [G, G, G, G]

    # perform non maximum suppression in gradient direction
    output_img = np.select(cond_list, choice_list, 0)

    return output_img
