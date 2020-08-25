import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
import imageio
import time
import cv2
import os
import warnings
warnings.filterwarnings("ignore")

matplotlib.use("Agg")

from multiview_calib import utils 
from multiview_calib.bundle_adjustment_scipy import (build_input, bundle_adjustment, evaluate, 
                                                     unpack_camera_params)
from multiview_calib.singleview_geometry import reprojection_error
from multiview_calib.calibration import verify_landmarks

__config__ = { 
    "ftol":1e-18,
    "xtol":1e-18,
    "loss":"linear",
    "f_scale":1,
    "max_nfev":200, # first optimization
    "bounds_cp":[0]*15,
    "bounds_pt":[1000]*3,    
    "output_path": "output/triangulate_image_points/"
}

def main(poses='poses.json',
         landmarks='landmarks.json',
         dump_images=True): 
    
    if dump_images:
        utils.mkdir(__config__["output_path"])

    poses = utils.json_read(poses)
    landmarks = utils.json_read(landmarks)
        
    intrinsics = {view:{'K':data['K'], 'dist':data['dist']} for view,data in poses.items()}
    extrinsics = {view:{'R':data['R'], 't':data['t']} for view,data in poses.items()}    
  
    res, msg = verify_landmarks(landmarks)
    if not res:
        raise ValueError(msg)    

    views = list(landmarks.keys())
    print("-"*20)
    print("Views: {}".format(views))
    
    print("Triangulate image points...(this can take a while depending on the number of points to triangulate)")
    start = time.time()
    camera_params, points_3d, points_2d,\
    camera_indices, point_indices, \
    n_cameras, n_points, ids = build_input(views, intrinsics, extrinsics, 
                                           landmarks, each=1, 
                                           view_limit_triang=4)
    print("Elapsed time: {:0.2f}s".format(time.time()-start))
        
    print("Least-Squares optimization of the 3D points:")
    print("\t ftol={:0.3e}".format(__config__["ftol"]))
    print("\t xtol={:0.3e}".format(__config__["xtol"]))
    print("\t loss={} f_scale={:0.2f}".format(__config__["loss"], __config__['f_scale']))
    print("\t max_nfev={}".format(__config__["max_nfev"]))
        
    points_3d_ref = bundle_adjustment(camera_params, points_3d, points_2d, camera_indices, 
                                     point_indices, n_cameras, n_points, 
                                     optimize_camera_params=False, 
                                     optimize_points=True, 
                                     ftol=__config__["ftol"], xtol=__config__["xtol"],
                                     loss=__config__['loss'], f_scale=__config__['f_scale'],
                                     max_nfev=__config__["max_nfev"], 
                                     bounds=True, 
                                     bounds_cp = __config__["bounds_cp"],
                                     bounds_pt = __config__["bounds_pt"],
                                     verbose=True, eps=1e-12)        

    f1 = evaluate(camera_params, points_3d_ref, points_2d, 
                  camera_indices, point_indices, 
                  n_cameras, n_points)

    avg_abs_res = np.abs(f1).mean()
    print("Average absolute residual: {:0.2f} over {} points.".format(avg_abs_res, len(f1)/2))
            
    if dump_images:
        plt.figure()
        plt.plot(f1)
        plt.title("Residuals after optimization")
        plt.ylabel("Residual [pixels]")
        plt.xlabel("X and Y coordinates")        
        plt.show()
        plt.savefig(os.path.join(__config__["output_path"], "optimized_residuals.jpg"), bbox_inches='tight')  
        plt.ylim(-10,10)
        plt.savefig(os.path.join(__config__["output_path"], "optimized_residuals_ylim.jpg"), bbox_inches='tight')

    print("Reprojection errors (mean+-std pixels):")
    for i,(view, cp) in enumerate(zip(views, camera_params)):
        
        K, R, t, dist = unpack_camera_params(cp)

        points3d = points_3d_ref[point_indices[camera_indices==i]]
        points2d = points_2d[camera_indices==i]
        
        mean_error, std_error = reprojection_error(R, t, K, dist, points3d, points2d, 'mean')
        print("\t {} n_points={}: {:0.3f}+-{:0.3f}".format(view, len(points3d), mean_error, std_error))
        
    print("Reprojection errors (median pixels):")
    for i,(view, cp) in enumerate(zip(views, camera_params)):
        
        K, R, t, dist = unpack_camera_params(cp)

        points3d = points_3d_ref[point_indices[camera_indices==i]]
        points2d = points_2d[camera_indices==i]
        
        mean_error, std_error = reprojection_error(R, t, K, dist, points3d, points2d, 'median')
        print("\t {} n_points={}: {:0.3f}".format(view, len(points3d), mean_error))        
        
    points = {"points_3d": points_3d_ref.tolist(), 
              "ids":np.array(ids).tolist()}  
    
    utils.json_write("triangulated_points.json", points)
    
if __name__ == "__main__":

    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')    

    parser = argparse.ArgumentParser()  
    parser.add_argument("--poses", "-p", type=str, required=True, default="poses.json",
                        help='JSON file containing the intrinsic and extrinsics parameters for each view')    
    parser.add_argument("--landmarks", "-l", type=str, required=True, default="landmarks.json",
                        help='JSON file containing the image landmarks for each view')
    parser.add_argument("--dump_images", "-d", default=False, const=True, action='store_const',
                        help='Saves images for visualisation') 

    args = parser.parse_args()

    main(**vars(args))

# python triangulate_image_points.py -p poses.json -l landmarks.json --dump_images 