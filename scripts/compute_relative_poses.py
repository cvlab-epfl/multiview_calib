import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib
import imageio
import cv2
import os
'''
matplotlib.use('GTK')
if os.environ.get('DISPLAY') is None:
    print("Cannot connect to display.")
    matplotlib.use('Agg')
'''
from multiview_calib import utils 
from multiview_calib.twoview_geometry import (compute_relative_pose, residual_error,
                                              sampson_distance, draw_epilines)

def main(setup='setup.json',
         intrinsics='intrinsics.json',
         landmarks='landmarks.json',
         filenames='filenames.json',         
         method='8point',
         dump_images=True,
         th=1):
    
    setup = utils.json_read(setup)
    intrinsics = utils.json_read(intrinsics)
    landmarks = utils.json_read(landmarks)    
    
    relative_poses = {}
    for view1, view2 in setup['minimal_tree']:

        landmarks1 = landmarks[view1]["landmarks"]
        landmarks2 = landmarks[view2]["landmarks"]    
        idxs1 = landmarks[view1]["timestamp"]
        idxs2 = landmarks[view2]["timestamp"] 

        idxs_common = set(idxs1).intersection(idxs2)

        pts1 = np.vstack([landmarks1[idxs1.index(idx)] for idx in idxs_common])
        pts2 = np.vstack([landmarks2[idxs2.index(idx)] for idx in idxs_common])

        K1 = np.float64(intrinsics[view1]['K'])
        K2 = np.float64(intrinsics[view2]['K'])
        dist1 = np.float64(intrinsics[view1]['dist'])
        dist2 = np.float64(intrinsics[view2]['dist'])

        try:
            Rd, td, F, pts1_undist, pts2_undist, tri = compute_relative_pose(pts1, pts2,
                                                                             K1=K1, dist1=dist1,
                                                                             K2=K2, dist2=dist2, 
                                                                             method=method, th=th)
        except:
            print("---------------> Exception: {},{}".format(view1, view2))
            raise
            
        print("Pair {}:".format([view1, view2]))
        print("\t Fundmanetal matrix:\n\t\t{}\n\t\t{}\n\t\t{}".format(F[0],F[1],F[2]))
        print("\t Right camera position:\n\t\t{}".format(utils.invert_Rt(Rd, td)[1].ravel()))
        print("\t Residual error: {}".format(residual_error(pts1_undist, pts2_undist, F)[0]))
        print("\t Sampson distance: {}".format(sampson_distance(pts1_undist, pts2_undist, F)[0]))         

        relative_poses[str((view1, view2))] = {"F":F.tolist(), "Rd":Rd.tolist(), "td":td.tolist(), 
                                          "triang_points":tri.tolist(), 
                                          "timestamps":list(idxs_common)}

        if dump_images:
            
            fnames = utils.json_read(filenames)
            
            img1 = imageio.imread(fnames[view1])
            img2 = imageio.imread(fnames[view2])  

            img1_undist = cv2.undistort(img1.copy(), K1, dist1, None, K1)
            img2_undist = cv2.undistort(img2.copy(), K2, dist2, None, K2)

            idx = np.arange(pts1_undist.shape[0])
            np.random.shuffle(idx)
            img1_, img2_ = draw_epilines(img1_undist, img2_undist, pts1_undist[idx[:50]], pts2_undist[idx[:50]],
                                         F, None, linewidth=2, markersize=20)

            utils.mkdir("output/relative_poses/")
            
            hmin = np.minimum(img1_.shape[0], img2_.shape[0])
            imageio.imsave("output/relative_poses/{}_{}.jpg".format(view1, view2), np.hstack([img1_[:hmin],img2_[:hmin]]))
            
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
    parser.add_argument("--method", "-m", type=str, required=False, default="8point",
                        help='Method to compute fundamental matrix: \'8point\', \'lmeds\' or \'ransac\'') 
    parser.add_argument("--filenames", "-f", type=str, required=False, default="filenames.json",
                        help='JSON file containing one filename of an image for each view. Used onyl if --dump_images is on')
    parser.add_argument("--dump_images", "-d", default=False, const=True, action='store_const',
                        help='Saves images for visualisation')
    parser.add_argument("--th", "-th", type=int, required=False, default=1,
                        help='Threshold for RANSAC method')    
    
    args = parser.parse_args()

    main(**vars(args))

# python compute_relative_poses.py -s setup.json -i intrinsics.json -l landmarks.json -f filenames.json --dump_images  