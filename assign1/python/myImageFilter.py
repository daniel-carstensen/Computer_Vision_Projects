# Author : Daniel Carstensen
# Date : 01/19/2023
# File name : myImageFilter.py
# Class : COSC83
# Purpose : Create function to convolve an image with a kernel

import numpy as np


def myImageFilter(img0, h):
    # pad image to allow for convolution at the edges
    pad_shape = np.floor(h.shape[0] / 2).astype(np.int16)
    img0 = np.pad(img0, pad_width=(pad_shape,), mode='edge')
    # create subarray of windows of kernel shape
    windows = np.lib.stride_tricks.sliding_window_view(img0, h.shape)
    # multiply all windows with flipped kernel and sum up to retrieve convolved image
    output_img = np.einsum('ij,klij->kl', np.flip(h), windows)

    return output_img
