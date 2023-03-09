import numpy as np
from scipy.interpolate import RectBivariateSpline


def LucasKanade(It, It1, rect):
    # Input: 
    #   It: template image
    #   It1: Current image
    #   rect: Current position of the object
    #   (top left, bot right coordinates: x1, y1, x2, y2)
    # Output:
    #   p: movement vector dx, dy

    # set up the threshold
    threshold = 0.01875
    maxIters = 100
    p = np.zeros(2)          
    x1, y1, x2, y2 = rect

    # put your implementation here
    x, y = np.arange(It.shape[1]), np.arange(It.shape[0])

    It_y, It_x = np.gradient(It1)

    It_ip = RectBivariateSpline(y, x, It)
    It1_ip = RectBivariateSpline(y, x, It1)
    It1_x_grad = RectBivariateSpline(y, x, It_x)
    It1_y_grad = RectBivariateSpline(y, x, It_y)

    # get the initial position of the object
    x, y = np.meshgrid(np.linspace(x1, x2, int(x2-x1)), np.linspace(y1, y2, int(y2-y1)))
    x, y = x.flatten(), y.flatten()
    template = np.array(It_ip.ev(y, x))

    for i in range(maxIters):
        x_warp, y_warp = np.linspace(x1 + p[0], x2 + p[0], int(x2-x1)), np.linspace(y1 + p[1], y2 + p[1], int(y2-y1))
        x_warp, y_warp = np.meshgrid(x_warp, y_warp)
        x_warp, y_warp = x_warp.flatten(), y_warp.flatten()

        It_warp = np.array(It1_ip.ev(y_warp, x_warp))

        b = template - It_warp

        x_grad = np.array(It1_x_grad.ev(y_warp, x_warp)).reshape(-1, 1)
        y_grad = np.array(It1_y_grad.ev(y_warp, x_warp)).reshape(-1, 1)
        grad = np.hstack([x_grad, y_grad])

        # in this case, the Jacobian is the identity, so we can ignore it
        dp = np.linalg.lstsq(grad, b, rcond=None)[0]
        p += dp

        if np.linalg.norm(dp) < threshold:
            break

    return p

