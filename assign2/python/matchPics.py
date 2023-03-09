import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection
from helper import plotMatches


def matchPics(I1, I2):
	# I1, I2 : Images to match
	# Convert Images to GrayScale
	gray1 = skimage.color.rgb2gray(I1)
	gray2 = skimage.color.rgb2gray(I2)
	
	# Detect Features in Both Images
	sigma = 0.1
	locs1 = corner_detection(gray1, sigma=sigma)
	locs2 = corner_detection(gray2, sigma=sigma)
	
	# Obtain descriptors for the computed feature locations
	desc1, locs1 = computeBrief(gray1, locs1)
	desc2, locs2 = computeBrief(gray2, locs2)

	# Match features using the descriptors
	ratio = 0.85
	matches = briefMatch(desc1, desc2, ratio=ratio)

	return matches, locs1, locs2
