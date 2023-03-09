import numpy as np
from scipy.interpolate import RectBivariateSpline


def InverseCompositionAffine(It, It1, rect):
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
    x1, y1, x2, y2 = rect
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # put your implementation here
    x_range = np.arange(It.shape[1])
    y_range = np.arange(It.shape[0])

    x, y = np.arange(x1, x2), np.arange(y1, y2)
    x, y = np.meshgrid(x, y)
    x, y = x.flatten(), y.flatten()

    y_grad, x_grad = np.gradient(It)
    y_grad, x_grad = y_grad[y1:y2, x1:x2].flatten(), x_grad[y1:y2, x1:x2].flatten()
    J = np.stack([x_grad * x, x_grad * y, x_grad, y_grad * x, y_grad * y, y_grad], axis=1)
    H = np.matmul(J.T, J)
    H_inv = np.linalg.pinv(H)

    It_ip = RectBivariateSpline(y_range, x_range, It)
    It1_ip = RectBivariateSpline(y_range, x_range, It1)

    for i in range(maxIters):
        M = np.array([[1, 0, 0], [0, 1, 0]])

        x_w = x * M[0, 0] + y * M[0, 1] + M[0, 2]
        y_w = x * M[1, 0] + y * M[1, 1] + M[1, 2]

        legal_coords = np.logical_and(np.logical_and(x_w >= x1, x_w < x2),
                                      np.logical_and(y_w >= y1, y_w < y2))

        x_valid = x[legal_coords]
        y_valid = y[legal_coords]
        x1_valid = x_w[legal_coords]
        y1_valid = y_w[legal_coords]
        J_valid = J[legal_coords, :]

        template = It_ip(y_valid, x_valid, grid=False)
        It1_w = It1_ip(y1_valid, x1_valid, grid=False)

        error = It1_w - template
        b = np.matmul(J_valid.T, error.flatten())

        dp = np.matmul(H_inv, b)

        p = np.array([[1 + dp[0], dp[1], dp[2]], [dp[3], 1 + dp[4], dp[5]], [0, 0, 1]])

        M = np.matmul(M, (np.linalg.pinv(p)))
        M = np.append(M, [[0, 0, 1]], axis=0)

        if np.linalg.norm(dp) < threshold:
            break

    return M[:2, :]


