import numpy as np
import cv2

from . import utils
from .singleview_geometry import undistort_points

__all__ = ["essential_to_fundamental", "fundamental_to_essential", "compute_right_epipole",
           "compute_left_epipole", "residual_error", "sampson_distance",
           "triangulate", "recover_pose", "compute_relative_pose",
           "draw_epilines", "visualise_cameras_and_triangulated_points", 
           "essential_from_relative_pose", "fundamental_from_relative_pose", 
           "relative_pose", "essential_from_poses", "fundamental_from_poses",
           "compute_epilines"]

def essential_to_fundamental(E, K1, K2):
    F = np.dot(np.linalg.inv(K2.T), np.dot(E, np.linalg.inv(K1)))
    F = F/F[2,2]    
    return F

def fundamental_to_essential(F, K1, K2):
    E = np.dot(K2.T, np.dot(F, K1))    
    E = E/E[2,2]
    return E

def compute_right_epipole(F):
    U, S, V = np.linalg.svd(F)
    e = V[-1]
    return e / e[2]

def compute_left_epipole(F):
    return compute_right_epipole(F.T)

def essential_from_relative_pose(Rd, td):
    tx = np.array([[0, -td[2], td[1]],
                   [td[2], 0, -td[0]],
                   [-td[1], td[0], 0]])

    E = np.dot(tx, Rd)  
    return E

def fundamental_from_relative_pose(Rd, td, K1, K2):
    E = essential_from_relative_pose(Rd, td)
    F = essential_to_fundamental(E, K1, K2)
    return F

def relative_pose(R1, t1, R2, t2):
    Rd = np.dot(R2, R1.T)
    td = t2-np.dot(Rd, t1)    
    return Rd, td

def essential_from_poses(K1, R1, t1, K2, R2, t2):
    Rd, td = relative_pose(R1, t1, R2, t2)
    E = essential_from_relative_pose(Rd, td)
    return E

def fundamental_from_poses(K1, R1, t1, K2, R2, t2):
    Rd, td = relative_pose(R1, t1, R2, t2)
    E = essential_from_relative_pose(Rd, td)
    F = essential_to_fundamental(E, K1, K2)
    return F

def distance_point_line(p, line):
    return np.abs(line[0]*p[0]+line[1]*p[1]+line[2])/np.sqrt(line[0]**2+line[1]**2)

def residual_error(pts1, pts2, F, mask=None):

    pts1_ = np.float64(pts1).reshape(-1,2)
    pts2_ = np.float64(pts2).reshape(-1,2)
    
    if mask is not None:
        pts1_ = pts1_[np.bool_(mask.ravel())]
        pts2_ = pts2_[np.bool_(mask.ravel())]
    
    lines2 = cv2.computeCorrespondEpilines(pts1_[:,None], 1, F).reshape(-1,3)
    lines1 = cv2.computeCorrespondEpilines(pts2_[:,None], 2, F).reshape(-1,3)
    
    errors = []
    for pt1, pt2, l1, l2 in zip(pts1_, pts2_, lines1, lines2): 
        errors.append((distance_point_line(pt1, l1) + distance_point_line(pt2, l2))/2)
        
    return np.mean(errors), errors

def sampson_distance(pts1, pts2, F, mask=None):
    
    pts1_ = np.float64(pts1).reshape(-1,2)
    pts2_ = np.float64(pts2).reshape(-1,2)
    F_ = F.astype(np.float64)    
    
    if mask is not None:
        pts1_ = pts1_[np.bool_(mask.ravel())]
        pts2_ = pts2_[np.bool_(mask.ravel())]    
    
    pts1_ = cv2.convertPointsToHomogeneous(pts1_).reshape(-1,3)
    pts2_ = cv2.convertPointsToHomogeneous(pts2_).reshape(-1,3)
    
    errors = []
    for pt1, pt2 in zip(pts1_, pts2_):         
        errors.append(cv2.sampsonDistance(pt1[None], pt2[None], F_))
        
    return np.mean(errors), errors

def triangulate(pts1, pts2, 
                K1=np.eye(3), R1=np.eye(3), t1=np.zeros(3), dist1=None, 
                K2=np.eye(3), R2=np.eye(3), t2=np.zeros(3), dist2=None):
    
    pts1_ = np.reshape(pts1, (-1,2)).copy()
    pts2_ = np.reshape(pts2, (-1,2)).copy()  
    
    if dist1 is not None and dist2 is not None:
        pts1_undist = undistort_points(pts1_, K1, dist1)
        pts2_undist = undistort_points(pts2_, K2, dist2)
    else:
        pts1_undist = pts1_
        pts2_undist = pts2_      
        
    P1 = np.dot(K1, np.hstack([R1, t1.reshape(3,1)]))
    P2 = np.dot(K2, np.hstack([R2, t2.reshape(3,1)]))

    tri = cv2.triangulatePoints(P1, P2, pts1_undist.T, pts2_undist.T)
    tri = tri[:3]/tri[3]
    tri = tri.T 
    
    return tri

def _triangulate(R, t, K1, K2, pts1_undist, pts2_undist):
    P1 = np.dot(K1, np.hstack([np.eye(3), np.zeros((3,1))]))
    P2 = np.dot(K2, np.hstack([R,t]))
    tri = cv2.triangulatePoints(P1, P2, np.float64(pts1_undist).T, np.float64(pts2_undist).T).T
    tri = tri[:,:3]/tri[:,[3]]  
    return tri

def recover_pose(E, K1, K2, pts1_undist, pts2_undist):
    
    Rx1, Rx2, tx = cv2.decomposeEssentialMat(E) 
    
    poses = [(Rx1, tx), (Rx1, -tx), (Rx2, tx), (Rx2,-tx)]
    n_in_front = []
    for R, t in poses:
        tri = _triangulate(R, t, K1, K2, pts1_undist, pts2_undist)
        n_in_front.append(np.sum(tri[:,2]>0))
        
    return poses[np.argmax(n_in_front)]

def compute_relative_pose(pts1, pts2, K1, dist1, K2=None, dist2=None, method='8point', th=0.1):
    
    if len(pts1)<8 or len(pts2)<8:
        raise ValueError("The algorithm needs at least 8 points!")
    
    if K2 is None:
        K2 = K1
        dist2 = dist1

    pts1 = np.asarray(pts1, np.float64) 
    pts2 = np.asarray(pts2, np.float64)
    K1 = np.asarray(K1, np.float64)
    K2 = np.asarray(K2, np.float64)    
    dist1 = np.asarray(dist1, np.float64)
    dist2 = np.asarray(dist2, np.float64)
    
    pts1_undist = undistort_points(pts1, K1, dist1)
    pts2_undist = undistort_points(pts2, K2, dist2)

    if method=='8point':
        F, mask = cv2.findFundamentalMat(pts1_undist, pts2_undist, cv2.FM_8POINT)
    elif method=='lmeds':
        F, mask = cv2.findFundamentalMat(pts1_undist, pts2_undist, cv2.FM_LMEDS)
    elif method=='ransac':
        F, mask = cv2.findFundamentalMat(pts1_undist, pts2_undist, cv2.FM_RANSAC, th)
    else:
        raise ValueError("Unrecognized method '{}'. Chose between '8point', 'lmeds' or 'ransac'".format(method))

    E = fundamental_to_essential(F, K1, K2)

    pts1_undist_masked = pts1_undist[np.bool_(mask).ravel()]
    pts2_undist_masked = pts2_undist[np.bool_(mask).ravel()]

    '''
    # recoverPose takes a single intrinsics matrix while we might have two different ones    
    # To this end we manually normalize the points in advance
    points1[:,0] = (points1[:,0] - K1[0,2]) / K1[1,1]
    points2[:,0] = (points2[:,0] - K2[0,2]) / K2[1,1]
    points1[:,1] = (points1[:,1] - K1[1,2]) / K1[2,2]
    points2[:,1] = (points2[:,1] - K2[1,2]) / K2[2,2]

    ret, R, t, mask2  = cv2.recoverPose(E, points1, points2, cameraMatrix=np.eye(3))
    t = t.reshape(3,1)

    # check if points are in front of the camera
    tri = _triangulate(R, t, K1, K2, pts1_undist, pts2_undist)
    
    p_points_front = np.mean(tri[:,2]>0)
    print("p_points_front:", p_points_front)
    if p_points_front<0.5:
        print("[info] {:.2f}% of points were not in front of the camera. We flipped t.".format((1-p_points_front)*100))
        #t = -t
        #tri = _triangulate(R, t, K1, K2, pts1_undist, pts2_undist)
    '''
    R, t = recover_pose(E, K1, K2, pts1_undist_masked, pts2_undist_masked)
    
    tri = _triangulate(R, t, K1, K2, pts1_undist, pts2_undist)
    
    return R, t, F, pts1_undist, pts2_undist, tri

def _draw_line(img, line, color=(255,0,0), linewidth=10):
    w = img.shape[1]
    x0,y0 = map(int, [0, -line[2]/line[1] ])
    x1,y1 = map(int, [w, -(line[2]+line[0]*w)/line[1] ])
    return cv2.line(img.copy(), (x0,y0), (x1,y1), tuple(color), linewidth)

def compute_epilines(pts1_undistorted, pts2_undistorted, F):
    
    if pts1_undistorted is None:
        lines2 = []
    else:
        pts1_ = np.float64(pts1_undistorted).reshape(-1,2)
        lines2 = cv2.computeCorrespondEpilines(pts1_[:,None], 1, F).reshape(-1,3)

    if pts2_undistorted is None:
        lines1 = []
    else:        
        pts2_ = np.float64(pts2_undistorted).reshape(-1,2)
        lines1 = cv2.computeCorrespondEpilines(pts2_[:,None], 2, F).reshape(-1,3)
    
    return lines1, lines2

def draw_epilines(img1_undistorted, img2_undistorted, 
                  pts1_undistorted, pts2_undistorted, 
                  F, mask=None, linewidth=10, markersize=10, scale=1):
    
    img1_, img2_ = img1_undistorted.copy(), img2_undistorted.copy()
    
    pts1_ = np.float64(pts1_undistorted).reshape(-1,2)
    pts2_ = np.float64(pts2_undistorted).reshape(-1,2)
    
    s = 1
    scale = np.array([[s,0,1], 
                      [0,s,1], 
                      [0,0,1]])    
    
    if mask is not None:
        pts1_ = pts1_[np.bool_(mask.ravel())]
        pts2_ = pts2_[np.bool_(mask.ravel())]
    
    lines2 = cv2.computeCorrespondEpilines(pts1_[:,None], 1, F).reshape(-1,3)
    for pt1, l2 in zip(pts1_, lines2): 
        color = tuple(int(x) for x in np.random.randint(0, 255, 3))
        
        img2_ = _draw_line(img2_, l2, color, linewidth)
        
        img1_ = cv2.circle(img1_, tuple(int(x) for x in pt1), radius=markersize, color=color, thickness=-1)
        img1_ = cv2.circle(img1_, tuple(int(x) for x in pt1), radius=markersize//2, color=(0,0,0), thickness=-1)        
        
    lines1 = cv2.computeCorrespondEpilines(pts2_[:,None], 2, F).reshape(-1,3)
    for pt2, l1 in zip(pts2_, lines1): 
        color = tuple(int(x) for x in np.random.randint(0, 255, 3))
        
        img1_ = _draw_line(img1_, l1, color, linewidth)

        img2_ = cv2.circle(img2_, tuple(int(x) for x in pt2), radius=markersize, color=color, thickness=-1)
        img2_ = cv2.circle(img2_, tuple(int(x) for x in pt2), radius=markersize//2, color=(0,0,0), thickness=-1)         
        
    return img1_, img2_

    
def visualise_cameras_and_triangulated_points(setup, poses, triang_points, max_points=100, path=None): 
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = ['r', 'b', 'g', 'y', 'k', 'm', 'c']

    all_points = []
    for i, (view1, view2) in enumerate(setup['minimal_tree']):

        R1 = np.asarray(poses[view1]['R'], np.float64)
        t1 = np.asarray(poses[view1]['t'], np.float64).reshape(3,1)
        R2 = np.asarray(poses[view2]['R'], np.float64)
        t2 = np.asarray(poses[view2]['t'], np.float64).reshape(3,1)    

        points_3d = np.asarray(triang_points[(view1, view2)]['triang_points'], np.float64).T
        
        # pick some point at random
        idxs = np.arange(points_3d.shape[1])
        np.random.shuffle(idxs)
        idxs = idxs[:max_points] 
        points_3d = points_3d[:,idxs]

        _, t1_inv = utils.invert_Rt(R1, t1)
        _, t2_inv = utils.invert_Rt(R2, t2)  

        all_points.append(t1_inv.reshape(1,3))
        all_points.append(t2_inv.reshape(1,3))
        all_points.append(points_3d.T)   

        if i<len(colors):
            color = colors[i]
        else:
            color = np.random.rand(3).tolist()

        ax.scatter(points_3d[0], points_3d[1], points_3d[2], c=color)
        ax.scatter(*t1_inv.ravel(), c=color, marker='x', s=250)
        ax.scatter(*t2_inv.ravel(), c=color, marker='s', s=100) 

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    all_points = np.vstack(all_points)
    x_min, y_min, z_min = np.min(all_points, axis=0)
    x_max, y_max, z_max = np.max(all_points, axis=0)

    ax.set_xlim(x_min-0.1*np.abs(x_min), x_max+0.1*np.abs(x_max))
    ax.set_ylim(y_min-0.1*np.abs(y_min), y_max+0.1*np.abs(y_max))
    ax.set_zlim(z_min-0.1*np.abs(z_min), z_max+0.1*np.abs(z_max))

    plt.show()
    
    if path is not None:
        utils.mkdir(path)
        ax.view_init(15, 0)
        plt.savefig(path+"/cameras_points3d_1.jpg", bbox_inches='tight')
        ax.view_init(15, 90)
        plt.savefig(path+"/cameras_points3d_2.jpg", bbox_inches='tight')
        ax.view_init(15+90, 0)
        plt.savefig(path+"/cameras_points3d_3.jpg", bbox_inches='tight')
        ax.view_init(15+90, 90)
        plt.savefig(path+"/cameras_points3d_4.jpg", bbox_inches='tight')
        ax.view_init(20, -125)
        plt.savefig(path+"/cameras_points3d_5.jpg", bbox_inches='tight')