import os
import sys
import json
import re
import os
import numpy as np
import cv2

__all__ = ["undistort_points", "invert_Rt", "project_points",
           "draw_points", "draw_rectangles"]

def undistort_points(points, K, distCoeffs, norm_coord=False):
    fx = K[0][0]
    fy = K[1][1]
    cx = K[0][2]
    cy = K[1][2]
    points_ = np.float32(points).reshape(-1,1,2)
    points_ = cv2.undistortPoints(points_, K, distCoeffs)
    points_ = np.reshape(points_, (-1,2))
    if not norm_coord:
        for p in points_:
            p[0] = p[0]*fx + cx
            p[1] = p[1]*fy + cy
    return points_

def project_points(pts, K, R, t, dist=None, image_shape=None):
    pts_ = np.reshape(pts, (-1,3))
    
    proj = np.dot(K, np.dot(R,pts_.T) + t.reshape(3,1))
    z = proj[2]
    xy = proj[:2].T/z[:,None]
    mask_in_front = z > 0  
    if image_shape is not None:
        mask_inside = np.logical_and.reduce([xy[:,0]>0, xy[:,0]<image_shape[1],
                                             xy[:,1]>0, xy[:,1]<image_shape[0]])
        mask_valid = np.logical_and(mask_in_front, mask_inside)
    else:
        mask_valid = mask_in_front
    
    rvec = cv2.Rodrigues(R)[0]
    proj = cv2.projectPoints(pts_, rvec, t, K, dist)[0].reshape(-1,2)

    return proj, mask_valid

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

def draw_points(image, centers, radius, color='r'): 
    """ Draws filled point on the image
    """
    _image = image.copy()        
    if color=='r':
        color = [255,0,0]
    elif color=='g':
        color = [0,255,0]
    elif color=='b':
        color = [0,0,255]
    elif color=='w':
        color = [255,255,255]
    elif color=='k':
        color = [0,0,0]
    
    for point in centers:
        _image = cv2.circle(_image, tuple(point.astype(np.int)), radius, color=color, thickness=-1)
    return _image

def draw_rectangles(image, centers, size, color='r', thickness=3): 
    """ Draws rectangles on the image
    """ 
    _image = image.copy()
    if color=='r':
        color = [255,0,0]
    elif color=='g':
        color = [0,255,0]
    elif color=='b':
        color = [0,0,255]
    elif color=='w':
        color = [255,255,255]
    elif color=='k':
        color = [0,0,0]
        
    for i, (x,y) in enumerate(np.int_(centers)):
        pt1 = (x-size[1]//2, y-size[0]//2)
        pt2 = (x+size[1]//2, y+size[0]//2)
        _image = cv2.rectangle(_image, pt1, pt2, color=color, thickness=thickness)
    return _image