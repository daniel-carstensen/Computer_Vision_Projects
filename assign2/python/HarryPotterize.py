import cv2
import skimage.io 
import skimage.color
# Import necessary functions
from matchPics import matchPics
from planarH import computeH_ransac, compositeH


# Write script for Q3.9
datadir = '../data'
cv_desk = cv2.imread('%s/cv_desk.png' % datadir)
cv_cover = cv2.imread('%s/cv_cover.jpg' % datadir)
hp_cover = cv2.imread('%s/hp_cover.jpg' % datadir)

matches, locs1, locs2 = matchPics(cv_cover, cv_desk)
H2to1, inliers = computeH_ransac(locs1[matches[:, 0]], locs2[matches[:, 1]])

hp_cover = cv2.resize(hp_cover, (cv_cover.shape[1], cv_cover.shape[0]))
composite_img = compositeH(H2to1, hp_cover, cv_desk)

cv2.imshow('Composite', composite_img)
cv2.imwrite('../figs/harryPotterized.png', composite_img)
cv2.waitKey(0)

