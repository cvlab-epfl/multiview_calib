import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
import imageio
import logging
import time
import cv2
import os
import warnings
warnings.filterwarnings("ignore")

matplotlib.use("Agg")

from multiview_calib import utils 
from multiview_calib.bundle_adjustment_scipy import (build_input, bundle_adjustment, evaluate, 
                                                     visualisation, unpack_camera_params)
from multiview_calib.singleview_geometry import reprojection_error
from multiview_calib.calibration import verify_view_tree, verify_landmarks

logger = logging.getLogger(__name__)

__config__ = {
    "each_training":1, # to use less datatpoint during the optimization
    "each_visualisation":1, # to use less datatpoints in the visualisation
    "optimize_camera_params":True, 
    "optimize_points":True, 
    "ftol":1e-8,
    "xtol":1e-8,
    "loss":"linear",
    "f_scale":1,
    "max_nfev":200, # first optimization
    "max_nfev2":200,# second optimization after outlier removal
    "bounds":True, 
    "bounds_cp":[0.3]*3+[1]*3+[10,10,10,10]+[0.01,0.01,0,0,0],
    "bounds_pt":[100]*3,
    "th_outliers_early":1000,
    "th_outliers":50, # value in pixels defining a point to be an outlier. If None, do not remove outliers.
    "output_path": "output/bundle_adjustment/"
}

def main(config=None,
         setup='setup.json',
         intrinsics='intrinsics.json',
         extrinsics='poses.json',
         landmarks='landmarks.json',
         filenames='filenames.json',
         iter1=200,
         iter2=200,
         dump_images=True):
    
    utils.config_logger(os.path.join(".", "bundle_adjustment.log"))

    if config is not None:
        __config__ = utils.json_read(config)
    
    if iter1 is not None:
        __config__["max_nfev"] = iter1
    if iter2 is not None:
        __config__["max_nfev2"] = iter2   
        
    if dump_images:
        utils.mkdir(__config__["output_path"])

    setup = utils.json_read(setup)
    intrinsics = utils.json_read(intrinsics)
    extrinsics = utils.json_read(extrinsics)
    landmarks = utils.json_read(landmarks)
    filenames_images = utils.json_read(filenames)
    
    if not verify_view_tree(setup['minimal_tree']):
        raise ValueError("minimal_tree is not a valid tree!")  
        
    res, msg = verify_landmarks(landmarks)
    if not res:
        raise ValueError(msg)    

    views = setup['views']
    logging.info("-"*20)
    logging.info("Views: {}".format(views))
    
    if __config__["each_training"]<2 or __config__["each_training"] is None:
        logging.info("Use all the landmarks.")
    else:
        logging.info("Subsampling the landmarks to 1 every {}.".format(__config__["each_training"]))    

    logging.info("Preparing the input data...(this can take a while depending on the number of points to triangulate)")
    start = time.time()
    camera_params, points_3d, points_2d,\
    camera_indices, point_indices, \
    n_cameras, n_points, ids, views_and_ids = build_input(views, intrinsics, extrinsics, 
                                                          landmarks, each=__config__["each_training"], 
                                                          view_limit_triang=4)
    logging.info("The preparation of the input data took: {:0.2f}s".format(time.time()-start))
    logging.info("Sizes:")
    logging.info("\t camera_params: {}".format(camera_params.shape))
    logging.info("\t points_3d: {}".format(points_3d.shape))
    logging.info("\t points_2d: {}".format(points_2d.shape))
    
    f0 = evaluate(camera_params, points_3d, points_2d,
                  camera_indices, point_indices,
                  n_cameras, n_points)    

    if dump_images:
        
        plt.figure()
        camera_indices_rav = np.vstack([camera_indices]*2).T.ravel()
        for view_idx in range(n_cameras):
            m = camera_indices_rav==view_idx
            plt.plot(f0[m], label='{}'.format(views[view_idx]))
        plt.title("Residuals at initialization")
        plt.ylabel("Residual [pixels]")
        plt.xlabel("X and Y coordinates")
        plt.legend()
        plt.grid()
        plt.show()
        plt.savefig(os.path.join(__config__["output_path"], "initial_residuals.jpg"), bbox_inches='tight')
        
    outliers = []
    if __config__["th_outliers_early"]==0 or __config__["th_outliers_early"] is None:
        logging.info("No early outlier rejection.")
    else:
        logging.info("Early Outlier rejection:")
        logging.info("\t threshold outliers: {}".format(__config__["th_outliers_early"])) 
        
        f0_ = np.abs(f0.reshape(-1,2))
        mask_outliers = np.logical_or(f0_[:,0]>__config__["th_outliers_early"],f0_[:,1]>__config__["th_outliers_early"])
        
        utils.json_write(os.path.join(__config__["output_path"], "outliers_early.json"), 
                         [views_and_ids[i]+(points_2d[i].tolist(),) for i,m in enumerate(mask_outliers) if m])        
        
        point_indices = point_indices[~mask_outliers]
        camera_indices = camera_indices[~mask_outliers]
        points_2d = points_2d[~mask_outliers]
        views_and_ids = [views_and_ids[i] for i,m in enumerate(~mask_outliers) if m]
        optimized_points = np.int32(list(set(point_indices)))
        logging.info("\t Number of points considered outliers: {}".format(sum(mask_outliers)))

        if sum(mask_outliers)/len(mask_outliers)>0.5:
            logging.info("!"*20)
            logging.info("More than half of the data points have been considered outliers! Something may have gone wrong.")
            logging.info("!"*20) 
        
    if dump_images:
        
        f01 = evaluate(camera_params, points_3d, points_2d,
                      camera_indices, point_indices,
                      n_cameras, n_points)         
        
        plt.figure()
        camera_indices_rav = np.vstack([camera_indices]*2).T.ravel()
        for view_idx in range(n_cameras):
            m = camera_indices_rav==view_idx
            plt.plot(f01[m], label='{}'.format(views[view_idx]))      
        plt.title("Residuals after early outlier rejection")
        plt.ylabel("Residual [pixels]")
        plt.xlabel("X and Y coordinates") 
        plt.legend()
        plt.grid()
        plt.show()
        plt.savefig(os.path.join(__config__["output_path"], "early_outlier_rejection_residuals.jpg"), bbox_inches='tight')        
        
    if __config__["bounds"]:
        logging.info("Bounded optimization:")
        logging.info("\t LB(x)=x-bound; UB(x)=x+bound")
        logging.info("\t rvec bounds=({},{},{})".format(*__config__["bounds_cp"][:3]))
        logging.info("\t tvec bounds=({},{},{})".format(*__config__["bounds_cp"][3:6]))
        logging.info("\t k bounds=(fx={},fy={},c0={},c1={})".format(*__config__["bounds_cp"][6:10]))
        logging.info("\t dist bounds=({},{},{},{},{})".format(*__config__["bounds_cp"][10:]))
        logging.info("\t 3d points bounds=(x={},y={},z={})".format(*__config__["bounds_pt"]))
    else:
        logging.info("Unbounded optimization.")
        
    logging.info("Least-Squares optimization of the 3D points:")
    logging.info("\t optimize camera parameters: {}".format(False))
    logging.info("\t optimize 3d points: {}".format(True))
    logging.info("\t ftol={:0.3e}".format(__config__["ftol"]))
    logging.info("\t xtol={:0.3e}".format(__config__["xtol"]))
    logging.info("\t loss={} f_scale={:0.2f}".format(__config__["loss"], __config__['f_scale']))
    logging.info("\t max_nfev={}".format(__config__["max_nfev"]))
        
    points_3d_ref = bundle_adjustment(camera_params, points_3d, points_2d, camera_indices, 
                                     point_indices, n_cameras, n_points, 
                                     optimize_camera_params=False, 
                                     optimize_points=True, 
                                     ftol=__config__["ftol"], xtol=__config__["xtol"],
                                     loss=__config__['loss'], f_scale=__config__['f_scale'],
                                     max_nfev=__config__["max_nfev"], 
                                     bounds=__config__["bounds"], 
                                     bounds_cp = __config__["bounds_cp"],
                                     bounds_pt = __config__["bounds_pt"], 
                                     verbose=True, eps=1e-12)        
        
    logging.info("Least-Squares optimization of 3D points and camera parameters:")
    logging.info("\t optimize camera parameters: {}".format(True))
    logging.info("\t optimize 3d points: {}".format(True)) 
    logging.info("\t ftol={:0.3e}".format(__config__["ftol"]))
    logging.info("\t xtol={:0.3e}".format(__config__["xtol"]))
    logging.info("\t loss={} f_scale={:0.2f}".format(__config__["loss"], __config__['f_scale']))
    logging.info("\t max_nfev={}".format(__config__["max_nfev"]))    
        
    new_camera_params, new_points_3d = bundle_adjustment(camera_params, points_3d_ref, points_2d, camera_indices, 
                                                         point_indices, n_cameras, n_points, 
                                                         optimize_camera_params=__config__["optimize_camera_params"], 
                                                         optimize_points=__config__["optimize_points"], 
                                                         ftol=__config__["ftol"], xtol=__config__["xtol"],
                                                         loss=__config__['loss'], f_scale=__config__['f_scale'],
                                                         max_nfev=__config__["max_nfev"], 
                                                         bounds=__config__["bounds"], 
                                                         bounds_cp = __config__["bounds_cp"],
                                                         bounds_pt = __config__["bounds_pt"], 
                                                         verbose=True, eps=1e-12)

    
    f1 = evaluate(new_camera_params, new_points_3d, points_2d, 
                  camera_indices, point_indices, 
                  n_cameras, n_points)

    avg_abs_res = np.abs(f1).mean()
    logging.info("Average absolute residual: {:0.2f} over {} points.".format(avg_abs_res, len(f1)/2))
    if avg_abs_res>15:
        logging.info("!"*20)
        logging.info("The average absolute residual error is high! Something may have gone wrong.".format(avg_abs_res))
        logging.info("!"*20)
            
    if dump_images:
        plt.figure()
        camera_indices_rav = np.vstack([camera_indices]*2).T.ravel()
        for view_idx in range(n_cameras):
            m = camera_indices_rav==view_idx
            plt.plot(f1[m], label='{}'.format(views[view_idx]))     
        plt.title("Residuals after optimization")
        plt.ylabel("Residual [pixels]")
        plt.xlabel("X and Y coordinates") 
        plt.legend()
        plt.grid()
        plt.show()
        plt.savefig(os.path.join(__config__["output_path"], "optimized_residuals.jpg"), bbox_inches='tight')        

    # Find ouliers points and remove them form the optimization.
    # These might be the result of inprecision in the annotations.
    # in this case we remove the resduals higher than 20 pixels.
    if __config__["th_outliers"]==0 or __config__["th_outliers"] is None:
        logging.info("No outlier rejection.")
    else:
        logging.info("Outlier rejection:")
        logging.info("\t threshold outliers: {}".format(__config__["th_outliers"])) 
        logging.info("\t max_nfev={}".format(__config__["max_nfev2"]))

        f1_ = np.abs(f1.reshape(-1,2))
        mask_outliers = np.logical_or(f1_[:,0]>__config__["th_outliers"],f1_[:,1]>__config__["th_outliers"])
        
        utils.json_write(os.path.join(__config__["output_path"], "outliers_optimized.json"), 
                         [views_and_ids[i]+(points_2d[i].tolist(),) for i,m in enumerate(mask_outliers) if m])
        
        point_indices = point_indices[~mask_outliers]
        camera_indices = camera_indices[~mask_outliers]
        points_2d = points_2d[~mask_outliers]
        views_and_ids = [views_and_ids[i] for i,m in enumerate(~mask_outliers) if m]
        optimized_points = np.int32(list(set(point_indices)))
        logging.info("\t Number of points considered outliers: {}".format(sum(mask_outliers)))
        

        
        
        if sum(mask_outliers)==0:
            logging.info("\t Exit.")
        else:
        
            if sum(mask_outliers)/len(mask_outliers)>0.5:
                logging.info("!"*20)
                logging.info("More than half of the data points have been considered outliers! Something may have gone wrong.")
                logging.info("!"*20)            

            logging.info("\t New sizes:")
            logging.info("\t\t camera_params: {}".format(camera_params.shape))
            logging.info("\t\t points_3d: {}".format(points_3d.shape))
            logging.info("\t\t points_2d: {}".format(points_2d.shape))
            
            if len(points_2d)==0:
                logging.info("No points left! Exit.")
                return

            new_camera_params, new_points_3d = bundle_adjustment(camera_params, points_3d_ref, points_2d, camera_indices, 
                                                                 point_indices, n_cameras, n_points, 
                                                                 optimize_camera_params=__config__["optimize_camera_params"], 
                                                                 optimize_points=__config__["optimize_points"], 
                                                                 ftol=__config__["ftol"], xtol=__config__["xtol"],
                                                                 loss=__config__['loss'], f_scale=__config__['f_scale'],
                                                                 max_nfev=__config__["max_nfev2"], 
                                                                 bounds=__config__["bounds"], 
                                                                 bounds_cp = __config__["bounds_cp"],
                                                                 bounds_pt = __config__["bounds_pt"], 
                                                                 verbose=True, eps=1e-12)


            f2 = evaluate(new_camera_params, new_points_3d, points_2d, 
                          camera_indices, point_indices, 
                          n_cameras, n_points)

            avg_abs_res = np.abs(f2).mean()
            logging.info("Average absolute residual: {:0.2f} over {} points.".format(avg_abs_res, len(f2)/2))
            if avg_abs_res>15:
                logging.info("!"*20)
                logging.info("The average absolute residual error (after outlier removal) is high! Something may have gone wrong.".format(avg_abs_res))
                logging.info("!"*20)

            if dump_images:
                plt.figure()
                camera_indices_rav = np.vstack([camera_indices]*2).T.ravel()
                for view_idx in range(n_cameras):
                    m = camera_indices_rav==view_idx
                    plt.plot(f2[m], label='{}'.format(views[view_idx]))
                plt.title("Residuals after outlier removal")
                plt.ylabel("Residual [pixels]")
                plt.xlabel("X and Y coordinates")    
                plt.legend()
                plt.grid()
                plt.show()
                plt.savefig(os.path.join(__config__["output_path"], "optimized_residuals_outliers_removal.jpg"),
                            bbox_inches='tight')

    logging.info("Reprojection errors (mean+-std pixels):")
    ba_poses = {}
    for i,(view, cp) in enumerate(zip(views, new_camera_params)):
        K, R, t, dist = unpack_camera_params(cp)
        ba_poses[view] = {"R":R.tolist(), "t":t.tolist(), "K":K.tolist(), "dist":dist.tolist()}
        
        points3d = new_points_3d[point_indices[camera_indices==i]]
        points2d = points_2d[camera_indices==i]
        
        mean_error, std_error = reprojection_error(R, t, K, dist, points3d, points2d)
        logging.info("\t {} n_points={}: {:0.3f}+-{:0.3f}".format(view, len(points3d), mean_error, std_error))
        
    logging.info("Reprojection errors (median pixels):")
    ba_poses = {}
    for i,(view, cp) in enumerate(zip(views, new_camera_params)):
        K, R, t, dist = unpack_camera_params(cp)
        ba_poses[view] = {"R":R.tolist(), "t":t.tolist(), "K":K.tolist(), "dist":dist.tolist()}
        
        points3d = new_points_3d[point_indices[camera_indices==i]]
        points2d = points_2d[camera_indices==i]
        
        mean_error, std_error = reprojection_error(R, t, K, dist, points3d, points2d, 'median')
        logging.info("\t {} n_points={}: {:0.3f}".format(view, len(points3d), mean_error))        
        
    ba_points = {"points_3d": new_points_3d[optimized_points].tolist(), 
                 "ids":np.array(ids)[optimized_points].tolist()}  
    
    if __config__["each_visualisation"]<2 or __config__["each_visualisation"] is None:
        logging.info("Visualise all the annotations.")
    else:
        logging.info("Subsampling the annotations to visualise to 1 every {}.".format(__config__["each_visualisation"]))    
        
    path = __config__['output_path'] if dump_images else None
    visualisation(setup, landmarks, filenames_images, 
                  new_camera_params, new_points_3d, 
                  points_2d, camera_indices, each=__config__["each_visualisation"], path=path)    

    utils.json_write("ba_poses.json", ba_poses)
    utils.json_write("ba_points.json", ba_points)
    
if __name__ == "__main__":

    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')    

    parser = argparse.ArgumentParser()  
    parser.add_argument("--config", "-c", type=str, required=False, default=None,
                        help='JSON file containing the config. parameters for the bundle adjusment')    
    parser.add_argument("--setup", "-s", type=str, required=True, default="setup.json",
                        help='JSON file containing the camera setup')
    parser.add_argument("--intrinsics", "-i", type=str, required=True, default="intrinsics.json",
                        help='JSON file containing the intrinsic parameters for each view')
    parser.add_argument("--extrinsics", "-e", type=str, required=True, default="extrinsics.json",
                        help='JSON file containing the extrinsic parameters for each view')    
    parser.add_argument("--landmarks", "-l", type=str, required=True, default="landmarks.json",
                        help='JSON file containing the landmarks for each view')
    parser.add_argument("--filenames", "-f", type=str, required=False, default="filenames.json",
                        help='JSON file containing one filename of an image for each view. Used onyl if --dump_images is on')

    parser.add_argument("--dump_images", "-d", default=False, const=True, action='store_const',
                        help='Saves images for visualisation') 
    
    parser.add_argument("--iter1", "-it1", type=int, required=False, default=None,
                        help='Maximum number of iterations of the first optimization')
    parser.add_argument("--iter2", "-it2", type=int, required=False, default=None,
                        help='Maximum number of iterations of the second optimization after outlier rejection')     
    
    args = parser.parse_args()

    main(**vars(args))

# python bundle_adjustment.py -s setup.json -i intrinsics.json -e extrinsics.json -l landmarks.json -f filenmaes.json --dump_images 