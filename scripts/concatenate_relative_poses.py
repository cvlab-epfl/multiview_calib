import numpy as np
import argparse
import matplotlib
import imageio
import cv2
import os

matplotlib.use("Agg")

from multiview_calib import utils 
from multiview_calib.point_set_registration import (estimate_scale_point_sets, procrustes_registration)
from multiview_calib.twoview_geometry import visualise_cameras_and_triangulated_points

def main(setup='setup.json',
         relative_poses='relative_poses.json',
         method='procrustes',
         dump_images=True):
    
    setup = utils.json_read(setup)
    relative_poses = utils.json_read(relative_poses) 
    relative_poses = utils.dict_keys_from_literal_string(relative_poses)
    
    # initialize the graph with the first pair of view
    # The first camera will be the center of our coordinate system for now
    pair0 = tuple(setup['minimal_tree'][0])
    poses = {pair0[0]: {"R":np.eye(3).tolist(), "t":np.zeros((3,1)).tolist()},
             pair0[1]: {"R":relative_poses[pair0]['Rd'],
                        "t":relative_poses[pair0]['td']}}
    triang_points = {pair0: {"triang_points":relative_poses[pair0]['triang_points'],
                             "timestamps":relative_poses[pair0]['timestamps']}}

    def find_adjacent_pair(pair):
        adj_pair = None
        inverse = False
        for key,data in triang_points.items():
            if pair[0] in key :
                adj_pair = key
            elif pair[1] in key:
                adj_pair = key
                inverse = True
                break 
        return adj_pair, inverse

    pairs = setup['minimal_tree'][1:]  

    while len(pairs)>0: 

        unmatched_pairs = []
        for curr_pair in pairs:
            curr_pair = tuple(curr_pair)

            adj_pair, inverse = find_adjacent_pair(curr_pair)

            if adj_pair is None:
                if curr_pair not in unmatched_pairs:
                    unmatched_pairs.append(curr_pair)
                continue

            if inverse:
                first_view = adj_pair[0]
                second_view = curr_pair[0]
            else:
                first_view = adj_pair[1]
                second_view = curr_pair[1]

            # this is the new 0,0,0 point for the current pair
            R1 = np.asarray(poses[first_view]['R'], np.float32)
            t1 = np.asarray(poses[first_view]['t'], np.float32).reshape(3,1)

            # relative pose of the current pair
            Rd = np.asarray(relative_poses[curr_pair]['Rd'], np.float64)
            td = np.asarray(relative_poses[curr_pair]['td'], np.float64).reshape(3,1)
            if inverse:
                Rd, td = utils.invert_Rt(Rd, td)

            # triangulated points of the adjacent pair
            p3d_adj = np.float64(triang_points[adj_pair]['triang_points'])
            idx_adj = triang_points[adj_pair]['timestamps']

            # triangulated points of the current pair
            p3d = np.float64(relative_poses[curr_pair]['triang_points'])
            idx = relative_poses[curr_pair]['timestamps']

            # find common points
            idx_common = set(idx_adj).intersection(idx)
            p3d_adj_com = np.array([p3d_adj[idx_adj.index(i)] for i in idx_common])
            p3d_com     = np.array([p3d[idx.index(i)] for i in idx_common])

            # estimate scale between the two pairs
            if method=='cross-ratios':
                relative_scale = estimate_scale_point_sets(p3d_com, p3d_adj_com)
            elif method=='procrustes':
                relative_scale,_,_,_ = procrustes_registration(p3d_com, p3d_adj_com)
            else:
                raise ValueError("Unrecognized method '{}'".format(method))

            # compute new camera pose for view2 of curr_pair
            R2 = np.dot(Rd, R1)
            t2 = np.dot(Rd, t1)+relative_scale*td   
            
            print("Pair: {}".format(curr_pair))
            print("\t Relative scale to {}: {:0.3f}".format(adj_pair, relative_scale))
            print("\t {} new position: {}".format(second_view, utils.invert_Rt(R2, t2)[1].ravel()))    

            # transform the triangulated points of the current pair to the origin
            if inverse:
                R_inv, t_inv = utils.invert_Rt(R2, t2)        
            else:
                R_inv, t_inv = utils.invert_Rt(R1, t1)
            p3d = np.dot(R_inv, np.float64(p3d).T*relative_scale)+t_inv

            poses[second_view] = {'R':R2.tolist(), 't':t2.tolist()}
            triang_points[curr_pair] = {'triang_points':p3d.T.tolist(),
                                        "timestamps":idx}

        if len(pairs)==len(unmatched_pairs):
            raise RuntimeError("The following pairs are not connected to the rest of the network: {}".format(unmatched_pairs))
            break

        pairs = unmatched_pairs[:]
        
    path = "output/relative_poses" if dump_images else None
    visualise_cameras_and_triangulated_points(setup, poses, triang_points, max_points=100,
                                              path=path)        
            
    utils.json_write("poses.json", poses)   
    triang_points = utils.dict_keys_to_string(triang_points)
    utils.json_write("triang_points.json", triang_points)
    
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
    parser.add_argument("--method", "-m", type=str, required=False, default="procrustes",
                        help='Method used to compute the relative scales between tow pairs of view: \'procrustes\' or \'cross-ratios\'') 

    parser.add_argument("--dump_images", "-d", default=False, const=True, action='store_const',
                        help='Saves images for visualisation')   
    
    args = parser.parse_args()

    main(**vars(args))

# python concatenate_relative_poses.py -s setup.json -r relative_poses.json --dump_images 