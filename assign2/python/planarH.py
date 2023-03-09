import numpy as np
import cv2
import scipy


def computeH(x1, x2):
	# Q3.6
	# Compute the homography between two sets of points
	x1 = np.array(x1)
	x2 = np.array(x2)
	A = None
	for i in range(x2.shape[0]):
		vec1 = np.concatenate((-1 * np.array(x2[i]), np.array([-1, 0, 0, 0]), np.array([x2[i, 0] * x1[i, 0], x2[i, 1] *
																						x1[i, 0], x1[i, 0]])))
		vec2 = np.concatenate((np.zeros(3), -1 * np.array(x2[i]), np.array([-1, x2[i, 0] * x1[i, 1], x2[i, 1] *
																			x1[i, 1], x1[i, 1]])))
		stacked = np.stack((vec1, vec2))

		if A is None:
			A = stacked
		else:
			A = np.vstack((A, stacked))

	u, s, vh = np.linalg.svd(A, full_matrices=True)
	h = vh[-1, :].T
	H2to1 = h.reshape((3, 3))

	return H2to1


def computeH_norm(x1, x2):
	# Q3.7
	# Compute the centroid of the points
	mean_x1 = np.mean(x1, axis=0)
	mean_x2 = np.mean(x2, axis=0)

	# Shift the origin of the points to the centroid
	x1_centrd = x1 - mean_x1
	x2_centrd = x2 - mean_x2

	# Normalize the points so that the largest distance from the origin is equal to sqrt(2)
	scale_x1 = np.sqrt(2) / np.amax(np.linalg.norm(x1_centrd, axis=1))
	scale_x2 = np.sqrt(2) / np.amax(np.linalg.norm(x2_centrd, axis=1))

	x1_tilde = np.multiply(scale_x1, x1_centrd)
	x2_tilde = np.multiply(scale_x2, x2_centrd)

	# Similarity transform 1
	T1 = np.array([scale_x1, 0, -scale_x1 * mean_x1[0], 0, scale_x1, -scale_x1 * mean_x1[1], 0, 0, 1]).reshape((3, 3))

	# Similarity transform 2
	T2 = np.array([scale_x2, 0, -scale_x2 * mean_x2[0], 0, scale_x2, -scale_x2 * mean_x2[1], 0, 0, 1]).reshape((3, 3))

	# Compute homography
	H2to1_norm = computeH(x1_tilde, x2_tilde)

	# Denormalization
	H2to1 = np.matmul(np.linalg.inv(T1), np.matmul(H2to1_norm, T2))

	return H2to1


def computeH_ransac(x1, x2):
	# Q3.8
	# Compute the best fitting homography given a list of matching points
	d = 0.25
	n = 2000
	max_inliers = np.zeros(x1.shape[0])
	bestH2to1 = None

	x1 = np.fliplr(x1)
	x2 = np.fliplr(x2)

	for i in range(n):
		inliers = np.zeros(x1.shape[0])
		indices = np.zeros(4)
		while len(indices) != len(set(indices)):
			indices = np.random.randint(0, x1.shape[0], 4)
		H2to1 = computeH_norm([x1[i] for i in indices], [x2[i] for i in indices])
		x2_homogeneous = np.hstack((x2, np.ones((x2.shape[0], 1))))
		x1_estimate = np.matmul(H2to1, x2_homogeneous.T)
		x1_estimate /= x1_estimate[2, :]

		error = np.linalg.norm(x1 - x1_estimate[:2, :].T, axis=1)
		inliers = np.where(error < d, 1, inliers)

		if np.sum(inliers) > np.sum(max_inliers):
			max_inliers = inliers
			bestH2to1 = H2to1

	return bestH2to1, max_inliers


def compositeH(H2to1, template, img):
	# Create a composite image after warping the template image on top
	# of the image using the homography

	# Note that the homography we compute is from the image to the template;
	# x_template = H2to1*x_photo
	# For warping the template to the image, we need to invert it.
	H2to1_inv = np.linalg.inv(H2to1)

	# Create mask of same size as template
	mask = np.ones(template.shape)

	# Warp mask by appropriate homography
	warped_mask = cv2.warpPerspective(mask, H2to1_inv, (img.shape[1], img.shape[0]))

	# Warp template by appropriate homography
	warped_template = cv2.warpPerspective(template, H2to1_inv, (img.shape[1], img.shape[0]))

	# Use mask to combine the warped template and the image
	composite_img = warped_template + img * np.logical_not(warped_mask)

	return composite_img
