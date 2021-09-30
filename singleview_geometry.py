import os
import sys
import json
import re
import os
import numpy as np
import cv2

__all__ = ["undistort_points", "invert_Rt", "project_points"]

def undistort_points(points, K, distCoeffs, norm_coord=False, newcameramtx=None):
    points_ = np.reshape(points, (-1,1,2))
    if newcameramtx is None:
        newcameramtx = K
    points_ = cv2.undistortPoints(np.float32(points_), K, distCoeffs, P=newcameramtx, R=None)
    points_ = np.reshape(points_, (-1,2))
    return points_

def project_points(pts, K, R, t, dist=None, image_shape=None):
    pts_ = np.array(pts)
    assert pts_.shape[1]==3
    
    proj = np.dot(K, np.dot(R,pts_.T) + t.reshape(3,1))
    z = proj[2]
    xy = proj[:2].T/z[:,None]
    mask_in_front = z > 0  
    
    rvec = cv2.Rodrigues(R)[0]
    proj = cv2.projectPoints(pts_, rvec, t, K, dist)[0].reshape(-1,2)
    
    if image_shape is not None:
        mask_inside = np.logical_and.reduce([proj[:,0]>0, proj[:,0]<image_shape[1],
                                             proj[:,1]>0, proj[:,1]<image_shape[0]])
        mask_valid = np.logical_and(mask_in_front, mask_inside)
    else:
        mask_valid = mask_in_front    

    return proj, mask_valid

def project_points_homography(H, points, return_mask=False, front_positive=True):
    """
    If `return_mask` is True, will return a mask indicating which points were
    projected "in front of" the camera.
    """

    points_shape = points.shape
    points = np.reshape(points, (-1, 2))

    p = np.vstack([points.T, np.ones(len(points))])
    transformed = np.dot(H, p)
    projected = transformed[:2] / transformed[2]

    projected = np.reshape(projected.T, points_shape)

    if np.linalg.det(H)<0:
        if front_positive:
            mask = transformed[2] >= 0
        else:
            mask = transformed[2] <= 0
    else:
        if front_positive:
            mask = transformed[2] <= 0
        else:
            mask = transformed[2] >= 0
    mask = np.reshape(mask, points_shape[:-1])

    if return_mask:
        return projected, mask

    return projected

def warpPerspectiveFrontal(src, M, dsize, borderValue=0):
    """
    This function is equivalent to OpenCV warpPerspective, but only points in
    front of the camera will be warped.
    """
    points = np.mgrid[:dsize[0], :dsize[1]].T
    M = np.linalg.inv(M)
    points, mask = project_points_homography(M, points, return_mask=True)

    dst = cv2.remap(src, np.float32(points), None, cv2.INTER_LINEAR, borderValue=borderValue)
    dst[np.logical_not(mask)] = borderValue

    return dst

def reprojection_error(R, t, K, dist, points3d, points2d, method='mean'):
    
    proj, mask_in_front = project_points(points3d, K, R, t, dist)
    
    distances = np.linalg.norm(points2d[mask_in_front]-proj[mask_in_front], axis=1)
    
    if method=='mean':
        return distances.mean(), distances.std()
    elif method=='median':
        return np.median(distances), distances.std()
    else:
        raise ValueError("Unrecognized method '{}'".format(method))

def change_intrinsics(points, K1, K2):
    fx1 = K1[0][0]
    fy1 = K1[1][1]
    cx1 = K1[0][2]
    cy1 = K1[1][2]
    
    fx2 = K2[0][0]
    fy2 = K2[1][1]
    cx2 = K2[0][2]
    cy2 = K2[1][2]    
    
    points_ = np.float32(points.copy()).reshape(-1,2)
    
    for p in points_:
        p[0] = (p[0] - cx1)/fx1
        p[1] = (p[1] - cy1)/fy1    
    
    for p in points_:
        p[0] = p[0]*fx2 + cx2
        p[1] = p[1]*fy2 + cy2
        
    return points_

def invert_Rt(R, t):
    Ri = R.T
    ti = np.dot(-Ri, t)
    return Ri, ti