import numpy as np
import argparse
import matplotlib
import imageio
import logging
import cv2
import os
import warnings
warnings.filterwarnings("ignore")

matplotlib.use("Agg")

from multiview_calib import utils
from multiview_calib.calibration import (compute_relative_poses_robust, visualise_epilines, 
                                            verify_view_tree, verify_landmarks)

logger = logging.getLogger(__name__)

def main(setup='setup.json',
         intrinsics='intrinsics.json',
         landmarks='landmarks.json',
         filenames='filenames.json',
         method='8point',
         th=20,
         max_paths=5,
         method_scale='cross-ratios',
         dump_images=True,
         output_path="output/relative_poses/"):
    
    utils.config_logger(os.path.join(".", "relative_poses_robust.log"))
    
    setup = utils.json_read(setup)
    intrinsics = utils.json_read(intrinsics)
    landmarks = utils.json_read(landmarks) 
    
    if not verify_view_tree(setup['minimal_tree']):
        raise ValueError("minimal_tree is not a valid tree!")
        
    res, msg = verify_landmarks(landmarks)
    if not res:
        raise ValueError(msg)        
    
    relative_poses = compute_relative_poses_robust(setup['views'], setup['minimal_tree'], intrinsics, 
                                                   landmarks, method=method, th=th, max_paths=max_paths,
                                                   verbose=1)
    
    if dump_images:
        visualise_epilines(setup['minimal_tree'], relative_poses, intrinsics, landmarks, 
                           filenames, output_path=output_path)
    
    relative_poses = utils.dict_keys_to_string(relative_poses)
    utils.json_write("relative_poses.json", relative_poses)

if __name__ == "__main__":

    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')    

    parser = argparse.ArgumentParser() 
    parser.add_argument("--setup", "-s", type=str, required=True, default="setup.json",
                        help='JSON file containing the camera setup')
    parser.add_argument("--intrinsics", "-i", type=str, required=True, default="intrinsics.json",
                        help='JSON file containing the intrinsics parameters')
    parser.add_argument("--landmarks", "-l", type=str, required=True, default="landmarks.json",
                        help='JSON file containing the landmark for each view')
    parser.add_argument("--filenames", "-f", type=str, required=False, default="filenames.json",
                        help='JSON file containing one filename of an image for each view. Used onyl if --dump_images is on')
    parser.add_argument("--method", "-m", type=str, required=False, default="lmeds",
                        help='Method to compute fundamental matrix: \'8point\', \'lmeds\' or \'ransac\'')
    parser.add_argument("--th", "-th", type=int, required=False, default=20,
                        help='Threshold for RANSAC method')  
    parser.add_argument("--max_paths", "-n", type=int, required=False, default=5,
                        help='Maximum number of paths from view1 to view2 to consider.')
    parser.add_argument("--method_scale", "-ms", type=str, required=False, default="cross-ratios",
                        help='Method used to compute the relative scales between tow pairs of view: \'procrustes\' or \'cross-ratios\'')    
    parser.add_argument("--dump_images", "-d", default=False, const=True, action='store_const',
                        help='Saves images for visualisation')
    
    args = parser.parse_args()

    main(**vars(args))

# python compute_relative_poses_robust.py -s setup.json -i intrinsics.json -l landmarks.json -m ransac --th 20 --dump_images
