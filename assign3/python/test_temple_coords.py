import numpy as np
import helper as hlp
import cv2
import submission as sub
import matplotlib.pyplot as plt

# 1. Load the two temple images and the points from data/some_corresp.npz
corresp = np.load('../data/some_corresp.npz')
I1 = cv2.imread('../data/im1.png')
I2 = cv2.imread('../data/im2.png')

I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2RGB)
I2 = cv2.cvtColor(I2, cv2.COLOR_BGR2RGB)

# 2. Run eight_point to compute F
F = sub.eight_point(corresp['pts1'], corresp['pts2'], max(I1.shape))

# 3. Load points in image 1 from data/temple_coords.npz
temple_coords = np.load('../data/temple_coords.npz')

# 4. Run epipolar_correspondences to get points in image 2
pts1 = temple_coords['pts1']
pts2 = sub.epipolar_correspondences(I1, I2, F, pts1)

# 5. Compute the camera projection matrix P1
intrinsics = np.load('../data/intrinsics.npz')
K1 = intrinsics['K1']
K2 = intrinsics['K2']
E = sub.essential_matrix(F, K1, K2)

P1 = np.matmul(K1, np.c_[np.identity(3), np.zeros((3, 1))])

# 6. Use camera2 to get 4 camera projection matrices P2
P2_candidates = hlp.camera2(E)
P2_1 = np.matmul(K2, P2_candidates[:, :, 0])
P2_2 = np.matmul(K2, P2_candidates[:, :, 1])
P2_3 = np.matmul(K2, P2_candidates[:, :, 2])
P2_4 = np.matmul(K2, P2_candidates[:, :, 3])
P2_candidates = [P2_1, P2_2, P2_3, P2_4]

# 7. Run triangulate using the projection matrices
pts3_candidates = [(sub.triangulate(P1, pts1, P2_1, pts2)), (sub.triangulate(P1, pts1, P2_2, pts2)),
                   (sub.triangulate(P1, pts1, P2_3, pts2)), (sub.triangulate(P1, pts1, P2_4, pts2))]

# 8. Figure out the correct P2
most_pos_depth = 0
best_pts = None
best_err = np.inf
for i in range(len(pts3_candidates)):
    if np.sum(pts3_candidates[i][0][:, -1] > 0) > most_pos_depth:
        if pts3_candidates[i][1] < best_err:
            P2 = P2_candidates[i]
            best_pts = pts3_candidates[i][0]
            best_err = pts3_candidates[i][1]
            most_pos_depth = np.sum(pts3_candidates[i][0][:, -1] > 0)

print(f'Error: {best_err}')


# 9. Scatter plot the correct 3D points
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(best_pts[:, 0], best_pts[:, 1], best_pts[:, 2])
plt.show()

# 10. Save the computed extrinsic parameters (R1,R2,t1,t2) to data/extrinsics.npz
extrinsics_1 = np.matmul(np.linalg.inv(K1), P1)
extrinsics_2 = np.matmul(np.linalg.inv(K2), P2)

R1 = extrinsics_1[:, :3]
t1 = extrinsics_1[:, -1]

R2 = extrinsics_2[:, :3]
t2 = extrinsics_2[:, -1]

np.savez('../data/extrinsics.npz', R1=R1, t1=t1, R2=R2, t2=t2)
