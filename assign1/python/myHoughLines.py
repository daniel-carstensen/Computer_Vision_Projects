# Author : Daniel Carstensen
# Date : 01/19/2023
# File name : myHoughLines.py
# Class : COSC83
# Purpose : Create function to read in Hough accumulator and return parameters of fitted lines

import numpy as np
import cv2  # For cv2.dilate function


def myHoughLines(H, nLines):
    # dilate accumulator to find maximum value in neighborhood (non maximum suppression)
    H_dil = cv2.dilate(H, np.ones((3, 3)))

    # suppress all other votes in neighborhood which likely resulted from noise
    H_nms = np.zeros((H.shape[0], H.shape[1]))
    H_nms = np.where(H == H_dil, H, H_nms)
    H_nms = H_nms.flatten()

    # find indices of nLines parameter-pairs with most votes
    idxs = np.argsort(H_nms)[-nLines:]
    rhos, thetas = np.unravel_index(idxs, H.shape)

    return rhos.astype(np.int64), thetas.astype(np.int64)
