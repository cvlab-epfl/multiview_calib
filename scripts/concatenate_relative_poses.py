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
from multiview_calib.extrinsics import (concatenate_relative_poses, visualise_cameras_and_triangulated_points, 
                                        verify_view_tree)

logger = logging.getLogger(__name__)

def main(setup='setup.json',
         relative_poses='relative_poses.json',
         method='cross-ratios',
         dump_images=True,
         output_path="output/relative_poses/"):
    
    utils.config_logger(os.path.join(output_path, "concat_relative_poses.log"))
    
    setup = utils.json_read(setup)
    relative_poses = utils.json_read(relative_poses) 
    relative_poses = utils.dict_keys_from_literal_string(relative_poses)
    
    if not verify_view_tree(setup['minimal_tree']):
        raise ValueError("minimal_tree is not a valid tree!")         
    
    poses, triang_points = concatenate_relative_poses(setup['minimal_tree'], relative_poses, method)
        
    path = output_path if dump_images else None
    visualise_cameras_and_triangulated_points(setup['views'], setup['minimal_tree'], poses, triang_points, 
                                              max_points=100, path=path)        
            
    utils.json_write(os.path.join(output_path, "poses.json"), poses)   
    triang_points = utils.dict_keys_to_string(triang_points)
    utils.json_write(os.path.join(output_path, "triang_points.json"), triang_points)
    
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
    parser.add_argument("--relative_poses", "-r", type=str, required=True, default="relative_poses.json",
                        help='JSON file containing the relative poses of each pair of view')
    parser.add_argument("--method", "-m", type=str, required=False, default="cross-ratios",
                        help='Method used to compute the relative scales between tow pairs of view: \'procrustes\' or \'cross-ratios\'') 

    parser.add_argument("--dump_images", "-d", default=False, const=True, action='store_const',
                        help='Saves images for visualisation')   
    
    args = parser.parse_args()

    main(**vars(args))

# python concatenate_relative_poses.py -s setup.json -r relative_poses.json --dump_images 
