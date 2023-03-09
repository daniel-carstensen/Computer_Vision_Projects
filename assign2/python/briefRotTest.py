import numpy as np
import matplotlib.pyplot as plt
import cv2
from matchPics import matchPics
import scipy
from helper import plotMatches


# Q3.5
# Read the image and convert to grayscale, if necessary
datadir = '../data'
img = cv2.imread('%s/cv_cover.jpg' % datadir)
num_matches = dict()
random_array = np.random.randint(1, 36, 3) * 10

for i in range(1, 36):
    # Rotate Image
    img_rot = scipy.ndimage.rotate(img, 360 - 10 * i)
    # Compute features, descriptors and Match features
    matches, locs1, locs2 = matchPics(img, img_rot)
    # Update histogram
    num_matches[str(360 - 10 * i)] = matches.shape[0]
    if (360 - 10 * i) in random_array:
        plotMatches(img, img_rot, matches, locs1, locs2)

# Display histogram
plt.bar(num_matches.keys(), num_matches.values())
plt.xticks(rotation=90)
plt.show()
