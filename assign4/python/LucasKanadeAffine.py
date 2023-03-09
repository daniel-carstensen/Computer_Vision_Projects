import numpy as np
from scipy.interpolate import RectBivariateSpline


def LucasKanadeAffine(It, It1, rect):
    # Input:
    #   It: template image
    #   It1: Current image
    #   rect: Current position of the object
    #   (top left, bot right coordinates: x1, y1, x2, y2)
    # Output:
    #   M: the Affine warp matrix [2x3 numpy array]

    # set up the threshold
    threshold = 0.01875
    maxIters = 100
    p = np.zeros(6)
    x1, y1, x2, y2 = rect

    # put your implementation here
    x_range = np.arange(It.shape[1])
    y_range = np.arange(It.shape[0])

    x, y = np.arange(x1, x2 + 1), np.arange(y1, y2 + 1)
    x, y = np.meshgrid(x, y)
    x, y = x.flatten(), y.flatten()

    It_ip = RectBivariateSpline(y_range, x_range, It)
    It1_ip = RectBivariateSpline(y_range, x_range, It1)

    for i in range(maxIters):
        M = np.array([[1 + p[0], p[1], p[2]], [p[3], 1 + p[4], p[5]]])

        x_w = x * M[0, 0] + y * M[0, 1] + M[0, 2]
        y_w = x * M[1, 0] + y * M[1, 1] + M[1, 2]

        legal_coords = np.logical_and(np.logical_and(x_w >= 0, x_w < It.shape[1]),
                                      np.logical_and(y_w >= 0, y_w < It.shape[0]))

        x_valid = x[legal_coords]
        y_valid = y[legal_coords]
        x1_valid = x_w[legal_coords]
        y1_valid = y_w[legal_coords]

        It1_x = It_ip(y1_valid, x1_valid, 0, 1, grid=False).flatten()
        It1_y = It1_ip(y1_valid, x1_valid, 1, 0, grid=False).flatten()

        J = np.stack([It1_x * x1_valid, It1_x * y1_valid, It1_x, It1_y * x1_valid, It1_y * y1_valid, It1_y], axis=1)
        H = np.matmul(J.T, J)

        template = It_ip(y_valid, x_valid, grid=False)
        It1_w = It1_ip(y1_valid, x1_valid, grid=False)

        error = template - It1_w
        b = np.matmul(J.T, error.flatten())

        dp = np.matmul(np.linalg.pinv(H), b)

        p += dp
        if np.linalg.norm(dp) < threshold:
            break

    return M

