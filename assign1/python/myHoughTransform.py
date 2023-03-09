# Author : Daniel Carstensen
# Date : 01/19/2023
# File name : myHoughTransform.py
# Class : COSC83
# Purpose : Create function to perform Hough transform on edge image

import numpy as np


def myHoughTransform(Im, rhoRes, thetaRes):
    # calculate maximum distance and scales of theta and rho based on input resolution
    rhoMax = np.ceil(np.sqrt(np.power(Im.shape[0], 2) + np.power(Im.shape[1], 2)))
    thetaScale = np.arange(0, 2*np.pi, thetaRes)
    rhoScale = np.arange(0, rhoMax, rhoRes)

    # find indices of nonzero elements in image
    Im = np.argwhere(Im > 0)

    # calculate vote matrix for parameters
    sinScale = np.sin(thetaScale)
    cosScale = np.cos(thetaScale)
    rhoVals = np.matmul(Im, np.array([sinScale, cosScale]))
    rhoVals = rhoVals.astype(np.int16)

    # count votes in accumulator through histogram with bins corresponding to the theta and rho scales
    H, thetaBins, rhoBins = np.histogram2d(np.tile(thetaScale, rhoVals.shape[0]), rhoVals.flatten(),
                                           [thetaScale, rhoScale])
    H = H.T     # transpose as instructed in numpy documentation

    return [H, rhoScale, thetaScale]
