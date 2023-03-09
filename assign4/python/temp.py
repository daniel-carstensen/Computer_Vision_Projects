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
    p = np.zeros((6, 1))
    x1, y1, x2, y2 = rect
    # print(rect)
    # put your implementation here
    x, y = np.arange(It1.shape[1]), np.arange(It1.shape[0])

    It_y, It_x = np.gradient(It1)

    It_ip = RectBivariateSpline(y, x, It)
    It1_ip = RectBivariateSpline(y, x, It1)
    It1_x_grad = RectBivariateSpline(y, x, It_x)
    It1_y_grad = RectBivariateSpline(y, x, It_y)

    # get the initial position of the object
    x, y = np.arange(x1, x2 + 1), np.arange(y1, y2 + 1)
    x, y = np.meshgrid(x, y)
    x, y = x.flatten(), y.flatten()

    x, y = x.reshape(-1, 1), y.reshape(-1, 1)
    pts_h = np.hstack([y, x, np.ones((x.shape[0], 1))])

    for i in tqdm(range(maxIters)):
        M = np.array([[1.0 + p[0], p[1], p[2]], [p[3], 1.0 + p[4], p[5]]]).reshape(2, 3)
        pts_h_warp = np.matmul(M, pts_h.T)

        y_warp = pts_h_warp[0].reshape(-1, 1)
        x_warp = pts_h_warp[1].reshape(-1, 1)

        coords_legal = np.logical_and(np.logical_and(x_warp >= 0, x_warp < It.shape[1]),
                                      np.logical_and(y_warp >= 0, y_warp < It.shape[0]))
        x_legal = x[coords_legal]
        y_legal = y[coords_legal]
        x_warp_legal = x_warp[coords_legal]
        y_warp_legal = y_warp[coords_legal]

        It_warp = np.array(It1_ip.ev(y_warp_legal, x_warp_legal)).reshape(-1, 1)
        template = np.array(It_ip.ev(y_legal, x_legal)).reshape(-1, 1)

        b = template - It_warp

        x_grad = np.array(It1_x_grad.ev(y_warp_legal, x_warp_legal)).reshape(-1, 1)
        y_grad = np.array(It1_y_grad.ev(y_warp_legal, x_warp_legal)).reshape(-1, 1)

        A = np.hstack([y_legal * y_grad, x_legal * y_grad, y_grad, y_legal * x_grad, x_legal * x_grad, x_grad])

        dp = np.linalg.lstsq(A, b, rcond=None)[0]

        p += dp

        if np.linalg.norm(dp) < threshold:
            break

    # reshape the output affine matrix
    M = np.array([[1.0 + p[0], p[1],    p[2]],
                 [p[3],     1.0 + p[4], p[5]]]).reshape(2, 3)

    return M


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

    # initialize affine transformation matrix M
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],[0,0,1]])

    tre_p=10
    p0=np.array([0,0,0,0,0,0])
    # gradient= np.asarray(np.gradient(It1))
    # X_gradient=gradient[1]
    # Y_gradient=gradient[0]
    H,W=It.shape
    X_range=np.arange(W)
    Y_range=np.arange(H)


    # extract top left and bottom right coordinates
    x1, y1, x2, y2 = rect
    X = np.arange(x1, x2 + 1)
    Y = np.arange(y1, y2 + 1)
    X, Y = np.meshgrid(X, Y)
    X = X.flatten()
    Y = Y.flatten()

    Spline=RectBivariateSpline(Y_range,X_range,It)
    Spline1=RectBivariateSpline(Y_range,X_range,It1)

    idx=0
    while tre_p>threshold and idx < maxIters:
        M=np.array([[1+p0[0], p0[1],    p0[2] ],
                   [p0[3],    1+p0[4],  p0[5] ],
                   [0,        0,        1.    ]])

        X_w=X*M[0,0]+Y*M[0,1]+M[0,2]
        Y_w=X*M[1,0]+Y*M[1,1]+M[1,2]

        mask_valid=np.logical_and(np.logical_and(X_w>=0,X_w<It.shape[1]),np.logical_and(Y_w>=0,Y_w<It.shape[0]))
        X_valid=X[mask_valid]
        Y_valid=Y[mask_valid]
        X1_valid=X_w[mask_valid]
        Y1_valid=Y_w[mask_valid]

        D_x=Spline1(Y1_valid,X1_valid,0,1,grid=False).flatten()
        D_y=Spline1(Y1_valid,X1_valid,1,0,grid=False).flatten()

        SDQ=np.stack([D_x*X1_valid,
                      D_x*Y1_valid,
                      D_x,
                      D_y*X1_valid,
                      D_y*Y1_valid,
                      D_y],axis=1)
        H = np.dot(np.transpose(SDQ),SDQ)

        template=Spline(Y_valid,X_valid,grid=False)
        It1_w=Spline1(Y1_valid,X1_valid,grid=False)

        Error=template-It1_w
        B= SDQ.T.dot(Error.flatten())
        delta_p =np.dot(np.linalg.inv(H),B)
        p0 = p0+delta_p
        tre_p = np.linalg.norm(delta_p)

        idx+=1

    return M[:2, :]

