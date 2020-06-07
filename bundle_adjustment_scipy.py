import numpy as np
import cv2
import os
from scipy.sparse import lil_matrix
import time
import imageio
import itertools
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

from . import utils
from .singleview_geometry import project_points, undistort_points
from .twoview_geometry import triangulate

__all__ = ["build_input", "bundle_adjustment", 
           "evaluate", "triangulate_all_pairs", "pack_camera_params", "unpack_camera_params",
           "visualisation", "error_measure"]

def unpack_camera_params(camera_params, rotation_matrix=True):

    r = np.float64(camera_params[:3])
    if rotation_matrix:
        r = cv2.Rodrigues(r)[0]
    t = np.float64(camera_params[3:6])
    K = np.float64([[camera_params[6],0,camera_params[8]],
                    [0,camera_params[7],camera_params[9]],
                    [0,0,1]])
    dist = np.float64(camera_params[10:15])  
    
    return K, r, t, dist

def pack_camera_params(K, R, t, dist):

    rvec = cv2.Rodrigues(np.float64(R))[0].ravel().tolist()
    tvec = np.float64(t).ravel().tolist()
    ks   = np.float64(K).ravel()[[0,4,2,5]].tolist()
    ds   = np.float64(dist).ravel().tolist()
    
    return rvec+tvec+ks+ds

def project(points, camera_params):
    """Convert 3-D points to 2-D by projecting onto images."""
    
    points_ = np.reshape(points, (-1,3))
    
    points_proj = []
    for p, cp in zip(points_, camera_params):

        K, rvec, tvec, dist = unpack_camera_params(cp, rotation_matrix=False)
        
        proj = cv2.projectPoints(p[None], rvec, tvec, K, dist)[0].reshape(1,2)
        points_proj.append(proj)
        
    points_proj = np.vstack(points_proj)
    return points_proj

def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
    """Compute residuals.
    
    `params` contains camera parameters and 3-D coordinates.
    """
    camera_params = params[:(n_cameras*15)].reshape((n_cameras, 15))
    points_3d = params[(n_cameras*15):].reshape((n_points, 3))

    points_proj = project(points_3d[point_indices], camera_params[camera_indices])
    
    return (points_proj - points_2d).ravel()

def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices,
                               optimize_camera_params=True, optimize_points=True):
    
    m = camera_indices.size * 2
    n = n_cameras * 15 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    
    if optimize_camera_params:
        for s in range(15):
            A[2 * i, camera_indices * 15 + s] = 1
            A[2 * i + 1, camera_indices * 15 + s] = 1

    if optimize_points:
        for s in range(3):
            A[2 * i, n_cameras * 15 + point_indices * 3 + s] = 1
            A[2 * i + 1, n_cameras * 15 + point_indices * 3 + s] = 1

    return A

def build_input(views, intrinsics, extrinsics, landmarks, each=1, view_limit_triang=2):

    timestamps = set().union(*[landmarks[view]['timestamp'] for view in views])
    timestamps = np.int32(list(timestamps)[::each])

    # camera parameters
    camera_params = [pack_camera_params(intrinsics[view]['K'], extrinsics[view]['R'],
                                        extrinsics[view]['t'], intrinsics[view]['dist']) 
                          for view in views]
    camera_params = np.float64(camera_params)

    # triangulate 3D positions from all possible pair of views
    points_3d_pairs = triangulate_all_pairs(views, landmarks, timestamps, camera_params, view_limit_triang)

    points_3d = []
    points_2d = []
    camera_indices = []
    point_indices = [] 
    n_cameras = len(views)

    n_points = 0    
    timestamps_kept = []
    for t, p3ds in zip(timestamps, points_3d_pairs):

        # find in which view sample i exists
        views_idxs = [j for j in range(n_cameras) if t in landmarks[views[j]]['timestamp']]

        if len(views_idxs)<2:
            continue

        # estimate of the 3d position
        p3d_mean = np.mean(np.reshape(p3ds, (-1,3)), axis=0)
        points_3d.append(p3d_mean)

        idxs = [landmarks[views[j]]['timestamp'].index(t) for j in views_idxs]               
        points_2d += [landmarks[views[j]]['landmarks'][idx] for j,idx in zip(views_idxs, idxs)]
        camera_indices += views_idxs  
        point_indices += [n_points]*len(views_idxs)
        timestamps_kept.append(t)

        n_points += 1

    point_indices = np.int32(point_indices)
    camera_indices = np.int32(camera_indices)
    points_3d = np.vstack(points_3d)
    points_2d = np.vstack(points_2d)
    
    return camera_params, points_3d, points_2d, camera_indices, \
           point_indices, n_cameras, n_points, timestamps_kept 

def bundle_adjustment(camera_params, points_3d, points_2d, 
                      camera_indices, point_indices, 
                      n_cameras, n_points, 
                      optimize_camera_params=True, optimize_points=True, 
                      ftol=1e-15, xtol=1e-15, max_nfev=200, loss='linear', f_scale=1, 
                      verbose=True, eps=1e-12, bounds=True, 
                      bounds_cp = [0.15]*6+[200,200,200,200]+[0.1,0.1,0,0,0],
                      bounds_pt = [100]*3):
    
    if optimize_camera_params==False and optimize_points==False:
        raise ValueError("One between 'optimize_camera_params' and 'optimize_points' should be True otherwise no variables are optimized!")

    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices,
                                   optimize_camera_params=optimize_camera_params,
                                   optimize_points=optimize_points)

    if verbose: t0 = time.time()
    x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))

    if bounds:
        if not optimize_camera_params:
            bounds_cp = [0]*15
        if not optimize_points:
            bounds_p = [0]*3            
        
        bounds = [[],[]]
        for x in camera_params:
            bounds[0] += (x-np.array(bounds_cp)-1e-12).tolist()
            bounds[1] += (x+np.array(bounds_cp)+1e-12).tolist() 

        for x in points_3d:  
            bounds[0] += (x-np.array(bounds_pt)-1e-12).tolist()
            bounds[1] += (x+np.array(bounds_pt)+1e-12).tolist()           
    else:
        bounds = (-np.inf, np.inf)
        if verbose:
            print("bounds (-inf, inf)")
        
    res = least_squares(fun, x0, jac='2-point', jac_sparsity=A, verbose=2 if verbose else 0, 
                        x_scale='jac', loss=loss, f_scale=f_scale, ftol=ftol, xtol=xtol, method='trf',
                        args=(n_cameras, n_points, camera_indices, point_indices, points_2d),
                        max_nfev=max_nfev, bounds=bounds)
    if verbose:
        t1 = time.time()
        print("Optimization took {0:.0f} seconds".format(t1 - t0))
    
    new_camera_params = res.x[:n_cameras*15].reshape(n_cameras, 15)
    new_points_3d = res.x[n_cameras*15:].reshape(n_points, 3)
    
    if optimize_camera_params and optimize_points:
        return new_camera_params, new_points_3d
    elif optimize_camera_params:
        return new_camera_params
    else:
        return new_points_3d
    
def evaluate(camera_params, points_3d, points_2d, 
             camera_indices, point_indices, 
             n_cameras, n_points):
    
    x = np.hstack((camera_params.ravel(), points_3d.ravel()))
    residuals = fun(x, n_cameras, n_points, camera_indices, point_indices, points_2d)    
    
    return residuals

def triangulate_all_pairs(views, landmarks, timestamps, camera_params, view_limit_triang=5):

    n_cameras = len(views)
    n_samples = len(timestamps)

    # to speed things up
    poses = []
    landmarks_undist = {}
    for j in range(n_cameras):
        K,R,t,dist = unpack_camera_params(camera_params[j])
        poses.append((K,R,t,dist))
        
        points = landmarks[views[j]]['landmarks']
        landmarks_undist[views[j]] = undistort_points(points, K, dist)
        
    '''
    triang_points = {}
    for j1, j2 in itertools.combinations(range(n_cameras), 2):

        K1,R1,t1,dist1 = poses[j1]
        K2,R2,t2,dist2 = poses[j2]

        ps1 = []
        ps2 = []
        for i in timestamps:
            timestamp1 = landmarks[views[j1]]['timestamp']
            timestamp2 = landmarks[views[j2]]['timestamp']
            if i in timestamp1 and i in timestamp2:
                i1 = timestamp1.index(i)
                i2 = timestamp2.index(i) 

                ps1.append(landmarks_undist[views[j1]][i1])
                ps2.append(landmarks_undist[views[j2]][i2])

        p3d = triangulate(ps1, ps2, K1, R1, t1, None, K2, R2, t2, None)  
        triang_points[(j1, j2)] = p3d
    '''
                     
    points_3d_pairs = []
    for i in timestamps:

        # find in which view sample i exists
        views_idxs = [j for j in range(n_cameras) if i in landmarks[views[j]]['timestamp']]
        
        # to estimate the 3D points we average the triangulations of 
        # multiple combinations of views. IF many cameras are used this might 
        # require a lot of time. TO limit the computational complexity
        # we limit here the number of views that are taken for the trinagulations
        views_idxs = views_idxs[:view_limit_triang]

        if len(views_idxs)<2:
            points_3d_pairs.append(None)
            continue 

        points_3d_pairs_ = []
        for j1, j2 in itertools.combinations(views_idxs, 2):

            K1,R1,t1,dist1 = poses[j1]
            K2,R2,t2,dist2 = poses[j2]

            i1 = landmarks[views[j1]]['timestamp'].index(i)
            i2 = landmarks[views[j2]]['timestamp'].index(i)            
            
            p3d = triangulate(landmarks_undist[views[j1]][i1], 
                              landmarks_undist[views[j2]][i2],
                              K1, R1, t1, None, K2, R2, t2, None)

            points_3d_pairs_.append(p3d)  
        points_3d_pairs.append(points_3d_pairs_) 
    return points_3d_pairs
            
def visualisation(setup, landmarks, filenames_images, camera_params, points_3d, points_2d, 
                  camera_indices, each=1, path=None):
    
    views = setup['views']
    
    timestamps = set().union(*[landmarks[view]['timestamp'] for view in views])
    timestamps = list(timestamps)[::each]    
    
    points_3d_tri = triangulate_all_pairs(views, landmarks, timestamps, camera_params)
    points_3d_tri_chained = np.vstack([item for sublist in points_3d_tri if sublist is not None 
                                            for item in sublist if item is not None])
    
    for idx_view, view in enumerate(views):    
            
        K, R, t, dist = unpack_camera_params(camera_params[idx_view])   
        
        proj_tri_pairs = project_points(points_3d_tri_chained, K, R, t, dist)

        proj = project_points(points_3d, K, R, t, dist)
        
        if view in filenames_images:
            image = imageio.imread(filenames_images[view])
        else:
            xmax, ymax = proj.max(axis=0)
            xmax = np.minimum(xmax, 5000)
            ymax = np.minimum(ymax, 3000)
            image = np.zeros((int(ymax), int(ymax)), np.uint8)

        def plot(p_tri, p_2d, prj, image, name):
            plt.figure(figsize=(10,8))
            plt.plot(p_tri[:,0], p_tri[:,1], 'k.', markersize=1, label='Triang. from pairs')            
            plt.plot(p_2d[:,0], p_2d[:,1], 'g.', markersize=10, label='Annotations')
            plt.plot(prj[:,0], prj[:,1], 'r.', markersize=5, label='B.A results')
            plt.imshow(image)
            plt.title("{}-{}".format(view, name))
            plt.legend()
            plt.show()
            if path is not None:
                utils.mkdir(path)
                plt.savefig(os.path.join(path, "ba_{}_{}.jpg".format(view, name)))
                
        plot(proj_tri_pairs, 
             points_2d[camera_indices==idx_view], 
             proj, 
             image, 
             "original")
            
        plot(undistort_points(proj_tri_pairs, K, dist), 
             undistort_points(points_2d[camera_indices==idx_view], K, dist),
             undistort_points(proj, K, dist),
             cv2.undistort(image, K, dist, None, K), 
             "undistorted")    
        
    # ------------ for 3D visualisation -----------   
    poses = {}
    for idx_view, view in enumerate(views):    
            
        K, R, t, dist = unpack_camera_params(camera_params[idx_view])   
        poses[view] = {"K":K.tolist(), "dist":dist.tolist(), "R":R.tolist(), "t":t.tolist()}
        
    triang_points = {}
    for i, (view1, view2) in enumerate(setup['minimal_tree']):
        triang_points[(view1, view2)] = {'triang_points':points_3d.tolist()}   
    
    from .twoview_geometry import visualise_cameras_and_triangulated_points
    visualise_cameras_and_triangulated_points(setup, poses, triang_points, 
                                              max_points=1000, path=path) 
    
def error_measure(setup, landmarks, ba_poses, ba_points, scale=1, view_limit_triang=5):
    
    views = setup['views']

    timestamps = ba_points['timestamp']
    points_3d = ba_points['points_3d']

    camera_params = []
    for view in views:
        camera_params.append(pack_camera_params(**ba_poses[view]))

    points_3d_tri = triangulate_all_pairs(views, landmarks, timestamps, camera_params, view_limit_triang=5)  

    avg_dists = []
    for p3d, tri in zip(points_3d, points_3d_tri):

        dist = np.linalg.norm(np.reshape(p3d,(1,3))-np.reshape(tri,(-1,3)), axis=1)*scale
        avg_dists.append(dist.mean())   

    return np.mean(avg_dists), np.std(avg_dists), np.median(avg_dists)