import numpy as np
import itertools

def estimate_scale_point_sets(src, dst, max_est=50000):
    
    idxs = np.arange(len(src))
    np.random.shuffle(idxs)
    
    # computes cross ratios between all pairs of points
    scales = []
    for i, (j,k) in enumerate(itertools.combinations(idxs, 2)):
        d1 = np.linalg.norm(src[j]-src[k])
        d2 = np.linalg.norm(dst[j]-dst[k])
        scales.append(d2/d1)
        
        if i>max_est:
            break
        
    return np.median(scales), np.std(scales)

def procrustes_registration(src, dst):
    """
    Estimates rotation translation and scale of two point sets
    using Procrustes analysis
    
    dst = (src*scale x R.T) + t + residuals
    
    Parameters:
    ----------
    src : numpy.ndarray (N,3)
        transformed points set
    dst : numpy.ndarray (N,3)
        target points set   
        
    Return:
    -------
    scale, rotation matrix, translation and average distance
    between the alligned points sets
    """
    from scipy.linalg import orthogonal_procrustes
    
    assert src.shape[0]==dst.shape[0]
    assert src.shape[1]==dst.shape[1]
    assert src.shape[1]==3    

    P = src.copy()
    Q = dst.copy()

    m1 = np.mean(P, 0) 
    m2 = np.mean(Q, 0)

    P -= m1
    Q -= m2

    norm1 = np.linalg.norm(P)
    norm2 = np.linalg.norm(Q)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Input matrices must contain >1 unique points")

    # change scaling of data (in rows) such that trace(mtx*mtx') = 1
    P /= norm1
    Q /= norm2
    R, s = orthogonal_procrustes(Q, P)
    
    scale = s*norm2/norm1
    t = m2-np.dot(m1*scale, R.T)
    
    src_trans = np.dot(src*scale, R.T) + t

    mean_distance = np.linalg.norm(src_trans-dst, axis=1).mean()
    
    return scale, R, t, mean_distance

def cloud_registration(src, dst, debug=False):
    from scipy import optimize
    import cv2
    import matplotlib.pyplot as plt
    assert src.shape[0] == dst.shape[0]
    assert src.shape[1] == dst.shape[1]
    assert src.shape[1] == 3

    def points_to_homogenous_coordinates(arr):
        assert arr.shape[1] == 3
        return np.append(arr.copy(), np.ones((arr.shape[0], 1)), 1)

    def RT_to_homogenous_coordinates(arr):
        assert arr.shape == (3, 4)
        return np.append(arr.copy(), [[0, 0, 0, 1]], 0)

    def average_distance(x, y):
        assert x.shape == y.shape
        delta = x - y
        return np.mean(np.sqrt(np.sum(delta * delta, axis=1)))

    def funct(params):
        a, b, c, d, e, f, g = params

        rvec = cv2.Rodrigues(np.array((a, b, c)))[0]
        tvec = np.array([d, e, f])
        rt = np.append(rvec, np.expand_dims(tvec, -1), 1)

        Q_estimated = g * np.matmul(rt, points_to_homogenous_coordinates(P).T).T

        return average_distance(Q_estimated, Q)

    def scale_rot_trans_decomposition(mat):
        assert mat.shape == (4, 4)
        sR = mat[0:3, 0:3]
        st = mat[0:3, 3]
        s = np.sqrt(np.matmul(sR, sR.T)[0, 0])

        return s, sR / s, st / s

    # # # Removing means # # #
    P = src.copy()
    m1 = np.mean(P, 0)
    P -= m1

    Q = dst.copy()
    m2 = np.mean(Q, 0)
    Q -= m2

    # # # Rescaling source# # #
    ratio_average_radiuses = np.sqrt(np.sum(np.max(Q, axis=0) ** 2)) / np.sqrt(np.sum(np.max(P, axis=0) ** 2))
    Rt_scaling = np.array([[ratio_average_radiuses, 0, 0, 0],
                           [0, ratio_average_radiuses, 0, 0],
                           [0, 0, ratio_average_radiuses, 0],
                           [0, 0, 0, 1]]
                          )

    P = np.matmul(Rt_scaling, points_to_homogenous_coordinates(P).T).T[:, 0:3]

    # # # Procrustes Registration # # #
    scale_procrustes, R_procrustes, t_procrustes, mean_distance_procrustes = procrustes_registration(P, Q)
    R_procrustes *= scale_procrustes
    t_procrustes *= scale_procrustes
    Rt_procrustes = RT_to_homogenous_coordinates(np.append(R_procrustes, np.expand_dims(t_procrustes, -1), 1))
    assert Rt_procrustes.shape == (4, 4)

    P = np.matmul(Rt_procrustes, points_to_homogenous_coordinates(P).T).T
    P = P[:, 0:3]

    if debug:
        print('Procrustes registration:', mean_distance_procrustes)

    # # # Optimization # # #
    to_opti = np.zeros((7,))
    param = optimize.minimize(funct, to_opti)

    a, b, c, d, e, f, g = param.x

    scale_optim = g
    R_optim = cv2.Rodrigues(np.array((a, b, c)))[0] * scale_optim
    t_optim = np.array([d, e, f]) * scale_optim
    Rt_optim = RT_to_homogenous_coordinates(np.append(R_optim, np.expand_dims(t_optim, -1), 1))
    assert Rt_optim.shape == (4, 4)

    Q_estimated = np.matmul(Rt_optim, points_to_homogenous_coordinates(P).T).T[:, 0:3]

    if debug:
        print('Optimization', average_distance(Q_estimated, Q))

    # # # Retrieving the overall transformation # # #
    R_m1 = np.identity(3)
    t_m1 = np.array(-1 * m1)  # substract
    Rt_m1 = RT_to_homogenous_coordinates(np.append(R_m1, np.expand_dims(t_m1, -1), 1))

    R_m2 = np.identity(3)
    t_m2 = np.array(m2)  # add
    Rt_m2 = RT_to_homogenous_coordinates(np.append(R_m2, np.expand_dims(t_m2, -1), 1))

    affine_transformation = np.matmul(Rt_m2,
                                      np.matmul(Rt_optim, np.matmul(Rt_procrustes, np.matmul(Rt_scaling, Rt_m1))))
    Q_without_decomposition = np.matmul(affine_transformation, points_to_homogenous_coordinates(src).T).T[:, 0:3]

    # # # Decomposition of the overall transformation into scale, rotation & translation # # #
    s, R, t = scale_rot_trans_decomposition(affine_transformation)

    affine_transformation[0:3, 0:3] = R
    affine_transformation[0:3, 3] = t

    Q_with_decomposition = s * np.matmul(affine_transformation, points_to_homogenous_coordinates(src).T).T[:, 0:3]

    if debug:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(Q_with_decomposition[:, 0], Q_with_decomposition[:, 1], Q_with_decomposition[:, 2], s=45,
                   label='with_decomposition')
        ax.scatter(Q_without_decomposition[:, 0], Q_without_decomposition[:, 1], Q_without_decomposition[:, 2], s=25,
                   label='without_decomposition')
        ax.scatter(dst[:, 0], dst[:, 1], dst[:, 2], s=15, label='q_final')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.legend()
        plt.show()
        print('Without scaling/rotation decomposition:', average_distance(Q_without_decomposition, dst))
        print('With scaling/rotation decomposition:', average_distance(Q_with_decomposition, dst))
        print('Distance between with and without scaling/rotation clouds',
              average_distance(Q_without_decomposition, Q_with_decomposition))
    return s, R, t, average_distance(Q_with_decomposition, dst)


if __name__ == '__main__':
    import cv2
    rot = cv2.Rodrigues(np.array([50, 32, 13], dtype=np.float32))[0]
    rt = np.append(rot, np.expand_dims(np.array([3, 100, -50]), -1), 1)
    c_src = np.random.standard_normal((100, 3))
    c_tar = (50 * np.matmul(rt, np.append(c_src.copy(), np.ones((100, 1)), 1).T)).T

    s, R, t, loss = cloud_registration(c_src, c_tar, debug=True)