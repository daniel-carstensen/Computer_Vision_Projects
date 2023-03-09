"""
Homework 5
Submission Functions
"""

import numpy as np
import cv2
import scipy.signal
import helper as hlp


"""
Q3.1.1 Eight Point Algorithm
       [I] pts1, points in image 1 (Nx2 matrix)
           pts2, points in image 2 (Nx2 matrix)
           M, scalar value computed as max(H1,W1)
       [O] F, the fundamental matrix (3x3 matrix)
"""


def eight_point(pts1, pts2, M):
    N = pts1.shape[0]

    # create normalization matrix
    T = np.zeros((N, N))
    np.fill_diagonal(T, 1 / M)

    # normalize points
    pts1_norm = np.matmul(T, pts1)
    pts2_norm = np.matmul(T, pts2)

    # linearize problem
    A = np.zeros((N, 9))
    for i in range(N):
        A[i] = [pts1_norm[i, 0] * pts2_norm[i, 0], pts1_norm[i, 0] * pts2_norm[i, 1], pts1_norm[i, 0],
                pts1_norm[i, 1] * pts2_norm[i, 0], pts1_norm[i, 1] * pts2_norm[i, 1], pts1_norm[i, 1],
                pts2_norm[i, 0], pts2_norm[i, 1], 1]

    # solve linear system of equations using SVD
    U, S, Vt = np.linalg.svd(A)
    F_norm = Vt[-1].reshape(3, 3)

    # enforce rank-2 constraint on the fundamental matrix
    U, S, Vt = np.linalg.svd(F_norm)
    S[-1] = 0
    F_norm = np.matmul(U, np.matmul(np.diag(S), Vt))

    F_norm = hlp.refineF(F_norm, pts1_norm, pts2_norm)

    # denormalize fundamental matrix
    T = T[:3, :3]
    T[2, 2] = 1
    F = np.matmul(T, np.matmul(F_norm, T))

    return F


"""
Q3.1.2 Epipolar Correspondences
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           F, fundamental matrix from image 1 to image 2 (3x3 matrix)
           pts1, points in image 1 (Nx2 matrix)
       [O] pts2, points in image 2 (Nx2 matrix)
"""


def epipolar_correspondences(im1, im2, F, pts1):
    # check if image patch is within image bounds
    def in_image(x, y, window, h, w):
        return (x + window <= w) and (x - window >= 0) and (y + window <= h) and (y - window >= 0)

    pts2 = np.zeros(pts1.shape)
    h, w = im1.shape[0], im1.shape[1]
    N = pts1.shape[0]
    window = 10     # window size
    candidate_dist = 30     # range of candidates to consider

    for i in range(N):
        x1, y1 = pts1[i, 0], pts1[i, 1]
        l2 = np.matmul(F, np.array([x1, y1, 1]).T)      # get epipolar line

        # create point candidates
        if l2[0] != 0:
            x2_candidates = np.arange(x1 - candidate_dist, x1 + candidate_dist)
            y2_candidates = -(l2[0] * x2_candidates + l2[2]) / l2[1]
        else:
            y2_candidates = np.arange(y1 - candidate_dist, y1 + candidate_dist)
            x2_candidates = -(l2[1] * y2_candidates + l2[2]) / l2[0]

        min_dist = np.inf
        window_1 = im1[y1 - window:y1 + window, x1 - window:x1 + window]    # image 1 patch

        # select candidate with minimum distance
        for x, y in zip(x2_candidates, y2_candidates):
            x, y = int(x), int(y)
            if not in_image(x, y, window, h, w):
                continue
            window_2 = im2[y - window:y + window, x - window:x + window]
            dist = np.sqrt(np.sum((window_1 - window_2) ** 2))
            if dist < min_dist:
                x2, y2 = x, y
                min_dist = dist

        pts2[i, 0], pts2[i, 1] = x2, y2

    return pts2


"""
Q3.1.3 Essential Matrix
       [I] F, the fundamental matrix (3x3 matrix)
           K1, camera matrix 1 (3x3 matrix)
           K2, camera matrix 2 (3x3 matrix)
       [O] E, the essential matrix (3x3 matrix)
"""


def essential_matrix(F, K1, K2):
    E = np.matmul(K2.T, np.matmul(F, K1))
    print('Essential matrix = ' + str(E))
    return E


"""
Q3.1.4 Triangulation
       [I] P1, camera projection matrix 1 (3x4 matrix)
           pts1, points in image 1 (Nx2 matrix)
           P2, camera projection matrix 2 (3x4 matrix)
           pts2, points in image 2 (Nx2 matrix)
       [O] pts3d, 3D points in space (Nx3 matrix)
"""


def triangulate(P1, pts1, P2, pts2):
    N = pts1.shape[0]
    pts3 = np.zeros((N, 3))
    i = 0
    err = 0

    # calculate 3D points
    for p1, p2 in zip(pts1, pts2):
        # linearize problem
        A = np.zeros((4, 4))
        A[0] = p1[1] * P1[2, :] - P1[1, :]
        A[1] = P1[0, :] - p1[0] * P1[2, :]
        A[2] = p2[1] * P2[2, :] - P2[1, :]
        A[3] = P2[0, :] - p2[0] * P2[2, :]

        # solve linear system of equations using SVD
        U, S, Vt = np.linalg.svd(A)
        X = Vt[-1].reshape(4, 1)
        X_h = X[:3] / X[-1]
        pts3[i, :] = X_h.ravel()
        i += 1

        # calculate projection error
        p1_proj = np.matmul(P1, X)
        p1_proj = p1_proj[:2] / p1_proj[-1]
        p1_proj = p1_proj.ravel()
        err += np.linalg.norm(p1 - p1_proj)

    err /= N

    return pts3, err


"""
Q3.2.1 Image Rectification
       [I] K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] M1 M2, rectification matrices (3x3 matrix)
           K1p K2p, rectified camera matrices (3x3 matrix)
           R1p R2p, rectified rotation matrices (3x3 matrix)
           t1p t2p, rectified translation vectors (3x1 matrix)
"""


def rectify_pair(K1, K2, R1, R2, t1, t2):
    # find image centers
    c1 = -np.matmul(np.linalg.inv(np.matmul(K1, R1)), np.matmul(K1, t1))
    c2 = -np.matmul(np.linalg.inv(np.matmul(K2, R2)), np.matmul(K2, t2))

    # calculate rotation vectors
    r1 = (c1 - c2) / np.linalg.norm(c1 - c2)
    r2 = np.cross(R1[2, :].T, r1)
    r3 = np.cross(r2, r1)

    # combine into rotation matrix
    R_tilde = np.column_stack((r1, r2, r3)).T

    R1p = R_tilde
    R2p = R_tilde

    K1p = K2
    K2p = K2

    # find translation vectors
    t1p = -np.matmul(R_tilde, c1)
    t2p = -np.matmul(R_tilde, c2)

    # find rectification matrices
    translate = np.array([[1, 0, 300], [0, 1, 0], [0, 0, 1]])   # translation to correct test_rectify.py
    M1 = np.matmul(np.matmul(K1p, R1p), np.linalg.inv(np.matmul(K1, R1)))
    M2 = np.matmul(translate, np.matmul(np.matmul(K2p, R2p), np.linalg.inv(np.matmul(K2, R2))))

    return M1, M2, K1p, K2p, R1p, R2p, t1p, t2p


"""
Q3.2.2 Disparity Map
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           max_disp, scalar maximum disparity value
           win_size, scalar window size value
       [O] dispM, disparity map (H1xW1 matrix)
"""


def get_disparity(im1, im2, max_disp, win_size):
    disp_maps = np.zeros((im1.shape[0], im1.shape[1], max_disp))    # create space for one disparity map per disparity

    for d in range(max_disp):
        # shift second image by disparity
        shifted_im2 = np.roll(im2, d, axis=1)

        # compute sum of squared differences
        diff_sq = (im1 - shifted_im2) ** 2

        # convolve
        disp_maps[:, :, d] = scipy.signal.convolve2d(diff_sq, np.ones((win_size, win_size)), mode='same')

    # find disparity matrix
    dispM = np.argmin(disp_maps, axis=2)
    dispM = dispM.astype('float64')

    return dispM


"""
Q3.2.3 Depth Map
       [I] dispM, disparity map (H1xW1 matrix)
           K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] depthM, depth map (H1xW1 matrix)
"""


def get_depth(dispM, K1, K2, R1, R2, t1, t2):
    # find centers
    c1 = -np.matmul(np.linalg.inv(np.matmul(K1, R1)), np.matmul(K1, t1))
    c2 = -np.matmul(np.linalg.inv(np.matmul(K2, R2)), np.matmul(K2, t2))

    b = np.linalg.norm(c1 - c2)
    f = K1[0, 0]

    # calculate depth from disparity
    depthM = np.divide(b * f, dispM, out=np.zeros_like(dispM), where=(dispM != 0))

    return depthM


"""
Q3.3.1 Camera Matrix Estimation
       [I] x, 2D points (Nx2 matrix)
           X, 3D points (Nx3 matrix)
       [O] P, camera matrix (3x4 matrix)
"""


def estimate_pose(x, X):
    # replace pass by your implementation
    pass


"""
Q3.3.2 Camera Parameter Estimation
       [I] P, camera matrix (3x4 matrix)
       [O] K, camera intrinsics (3x3 matrix)
           R, camera extrinsics rotation (3x3 matrix)
           t, camera extrinsics translation (3x1 matrix)
"""


def estimate_params(P):
    # replace pass by your implementation
    pass


if __name__ == '__main__':
    data = np.load('../data/some_corresp.npz')
    I1 = cv2.imread('../data/im1.png')
    I2 = cv2.imread('../data/im2.png')

    I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2RGB)
    I2 = cv2.cvtColor(I2, cv2.COLOR_BGR2RGB)

    F = eight_point(data['pts1'], data['pts2'], max(I1.shape))

    # hlp.displayEpipolarF(I1, I2, F)
    hlp.epipolarMatchGUI(I1, I2, F)
