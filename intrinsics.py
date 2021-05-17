import os
import sys
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)

def distortion_function(points_norm, dist):
    """
    Standard (OpenCV convention) distortion function
    
    Parameters
    ----------
    points_norm : numpy.ndarray (N,2)
        undistorted image points in normalized image coordinates.
        In other words, 3D object points transformed with [R,t]
    dist: list or numpy.ndarray (5,)
        distortion coefficients
    
    Return
    ------
    numpy.ndarray (N,2) distorted points
    """
    k_ = dist.reshape(5)

    x,y = points_norm[:,0], points_norm[:,1]
    r2 = x*x + y*y
    r4 = r2*r2
    r6 = r4*r2
    a1 = 2*x*y
    a2 = r2 + 2*x*x
    a3 = r2 + 2*y*y
    cdist = 1 + k_[0]*r2 + k_[1]*r4 + k_[4]*r6
    xd0 = x*cdist + k_[2]*a1 + k_[3]*a2
    yd0 = y*cdist + k_[2]*a3 + k_[3]*a1 

    return np.vstack([xd0, yd0]).T

def is_distortion_function_monotonic(dist, range=(0, 1.5, 100)):
    
    x = np.linspace(*range)
    px = distortion_function(np.vstack([x,x]).T, dist)[:,0]

    return np.all((px[1:]-px[:-1])>0) 

def enforce_monotonic_distortion(dist, K, image_points, proj_undist_norm,
                                 range_constraint=(0, 1.4, 1000), verbose=True):
    """
    Forces the distortion function to be monotonic in the given range.
    The range is defined in the normalized image coordinate system.
    It starts at the principal point and moves away from it.
    
    IMPORTANT: The monotonicity is enforce to the detriment of the accuracy of the calibration.
    A large range will induce a higher error. 
    Before using this, try to sample more precise points on the corner of the image first.
    If it is not enought switch to the Rational Camera Model.  

    Parameters
    ----------
    dist: list or numpy.ndarray (5,)
        initial distortion coefficients
    K: numpy.ndarray (3,3)
        intrinsic matrix
    image_points_norm : numpy.ndarray (N,2)
        image points (distorted) in normalized image coordinates
    proj_undist_norm : numpy.ndarray (N,2)
        projected object points (undistorted) in normalized image coordinates
    range_constraint : tuple (3,)
        range where the monotonicity must be enforced
    Return
    ------
    numpy.ndarray (5, ) new distortion coefficients
    """
    from scipy.optimize import minimize

    def diffs(points, k):
        proj = distortion_function(points,k)
        return proj[1:,:]-proj[:-1,:]

    # these are the points we want to be monotonous after undistorting them
    x_constraint = np.linspace(*range_constraint)
    x_constraint = np.vstack([x_constraint, x_constraint]).T

    def f(k_new):
        image_points_undist = cv2.undistortPoints(image_points, K, k_new).reshape(-1,2)
        cost = np.sum((image_points_undist-proj_undist_norm)**2, axis=1).mean()
        return cost

    def ineq_constraint(k_new):
        return diffs(x_constraint, k_new)[:,0]

    con = {'type': 'ineq', 'fun': ineq_constraint, 'lb':0, 'ub':None}

    x0 = dist.copy().reshape(5,)+0
    #bounds=[(x-np.abs(x)*0.1, x+np.abs(x)*0.1) for x in x0]
    bounds=[(x-np.abs(1e-6), x+np.abs(1e-6)) for x in x0[:-1]]+[(x0[-1]-1, x0[-1]+1)] # we only chnage k3
    
    res = minimize(f, x0, method='SLSQP', tol=1e-32, constraints=con, bounds=bounds,
                   options={'ftol': 1e-32, 'eps': 1e-12, 'disp': verbose, 'maxiter':1000})
    if verbose:
        print(res)

    new_dist = res.x

    if not is_distortion_function_monotonic(new_dist, range_constraint):
        #raise RuntimeError("Enforce monotonic distortion unsuccessful!")
        s = "Enforce monotonic distortion is unsuccessful"
        s += " but it does not mean that the distortion parameter are bad."
        print(s)

    return new_dist  

def probe_monotonicity(K, dist, newcameramtx, image_shape, N=100, M=100):
    
    # calculate the region in which to probe the monotonicity
    pts_undist = np.array([
        [0,0],
        [0,image_shape[0]],
        [image_shape[1],0],
        [image_shape[1], image_shape[0]]
    ])
    pts_norm = (pts_undist-newcameramtx[[0,1],[2,2]][None])/newcameramtx[[0,1],[0,1]][None]

    xmin, ymin = pts_norm.min(0)
    xmax, ymax = pts_norm.max(0)
    r_max = np.sqrt(xmax**2+ymax**2)

    # create points used to compute the sign after distortion
    alphas = np.linspace(0,np.pi/2, N//4+2)[1:-1]
    alphas = np.concatenate([alphas, alphas+np.pi/2, alphas+np.pi, alphas+np.pi*3/2])
    
    ds = r_max/M

    ptss = []
    sign = []
    for r in np.linspace(0, r_max, M):
        pts= np.vstack([r*np.cos(alphas), r*np.sin(alphas)]).T
        ptsp = np.vstack([(r+ds)*np.cos(alphas), (r+ds)*np.sin(alphas)]).T

        mask1 = np.logical_and(pts[:,0]>=xmin, pts[:,0]<xmax)
        mask2 = np.logical_and(pts[:,1]>=ymin, pts[:,1]<ymax)
        mask = np.logical_and(mask1, mask2)

        if np.all(mask==False):
            continue

        pts, ptsp = pts[mask],ptsp[mask]

        ptss.append((pts,ptsp))
        sign.append(np.sign(pts-ptsp))
        
    # distort the points
    grid, gridp = zip(*ptss)
    grid, gridp = np.vstack(grid), np.vstack(gridp)

    grid_ = np.hstack([grid, np.zeros((len(grid),1))])
    gridp_ = np.hstack([gridp, np.zeros((len(gridp),1))])

    proj1 = cv2.projectPoints(grid_, np.eye(3), np.zeros(3), np.eye(3), dist)[0].reshape(-1,2)
    proj2 = cv2.projectPoints(gridp_, np.eye(3), np.zeros(3), np.eye(3), dist)[0].reshape(-1,2)

    # probe 
    is_monotonic = np.sign(proj1-proj2)==np.vstack(sign)
    is_monotonic = np.logical_and(*is_monotonic.T)
    
    return grid, is_monotonic