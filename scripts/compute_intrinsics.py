import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import os
import datetime
import sys
import inspect
import argparse
import multiprocessing
import json
import pickle
import glob
import re
import imageio
import shutil
from itertools import repeat

from multiview_calib import utils
from multiview_calib.intrinsics import (enforce_monotonic_distortion, 
                                        is_distortion_function_monotonic,
                                        probe_monotonicity)

(cv2_major, cv2_minor, _) = cv2.__version__.split(".")
if int(cv2_major)<4:
    raise ImportError("Opencv version 4+ required!")

'''
ffmpeg -i VIDEO -r 0.5 frames/frame_%04d.jpg
python compute_intrinsics.py --folder_images ./frames -ich 6 -icw 8 -s 30 -t 24 --debug
https://markhedleyjones.com/storage/checkerboards/Checkerboard-A4-30mm-8x6.pdf
'''

__author__ = "Leonardo Citraro"
__email__ = "leonardo.citraro@epfl.ch"

def process_image(filename_image, inner_corners_height, inner_corners_width, debug, debug_folder):
    print("Processing image {} ...".format(filename_image))

    gray = utils.rgb2gray(imageio.imread(filename_image))

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (inner_corners_height,inner_corners_width),
                                             cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

    # If found, add object points, image points (after refining them)
    if not ret:
        return None
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    imgp = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

    if debug:
        gray = cv2.drawChessboardCorners(gray, (inner_corners_height,inner_corners_width), imgp, ret)
        imageio.imsave(os.path.join(debug_folder, os.path.basename(filename_image)), gray)

    return np.float32(imgp)

def main(folder_images, output_folder, description, 
         inner_corners_height, inner_corners_width, square_sizes, 
         alpha, threads, force_monotonicity, monotonic_range, 
         rational_model, fix_pp, intrinsic_guess, debug):
    """Main function for computing intrisic parameters.
    
    How to acquire the video of the checkerboard:
    1) Make sure the checkerboard is completely flat.
    2) Always keep the checkerboard at an angle w.r.t the camera plane.
       Images of checkerbaords whose planes are parallel to the camera plane are not useful.
    3) Keep the checkerboard close to the camera. The area of the checkerboard should intuitively be
       half of the area of the entire image.
    4) Cover the corners of the image as well. It is fine if the checkerboard goes outside
       the image; the algorithm will discard these automatically.

    Parameters
    ----------
    folder_images : str
        Path to the folder containing the images
    output_folder : str
        Path to the folder for storing the calibration results
    description : str
        Optional text that will be added to the json file
    inner_corners_height : int
        Number of corners the checkerboard has in the height dimention
    inner_corners_width : int
        Number of corners the checkerboard has in the width dimention
    square_sizes : float
        Size of the side of the squares the checkerboard uses
    rational_model : bool
        Use a camera model more suited for wider angle lenses.
        It computes 8 distortion parameters instead of the usual 5.
    alpha : float
        Free scaling parameter between 0 (when all the pixels in the undistorted image are valid)
        and 1 (when all the source image pixels are retained in the undistorted image)
    force_monotonicity: bool
        Forces monotonicity in the range defined by monotonic_range. To be used only if nothing else work.
    monotonic_range : float
        If greater than zero, it forces the distortion parameter to be monotonic in this range.
        To be used carefully.
    intrinsic_guess : str (optional)
        JSON file containing guesses for the intrinsic and distortion parameters.
    fix_pp : bool
        If True, freeze the pricipal point such that it is not optimized.
    threads : int
        Number of child processes that will be used to parallelize the detection phase.
    debug : bool
        If True, images and extra information is saved or printed.

    """
    
    debug_folder = os.path.join(output_folder, "debug")
    undistorted_folder = os.path.join(output_folder, "undistorted")

    # delete if exist
    utils.rmdir(debug_folder)
    utils.rmdir(undistorted_folder)

    utils.mkdir(undistorted_folder)
    if debug:
        utils.mkdir(debug_folder)

    print("-" * 50)
    print("Input parameters")
    print("-" * 50)
    print("folder_images:", folder_images)
    print("output_folder:", output_folder)
    print("description:", description)
    print("inner_corners_height:", inner_corners_height)
    print("inner_corners_width:", inner_corners_width)
    print("square_sizes:", square_sizes)
    print("rational_model:", rational_model)
    print("alpha:", alpha)
    print("force_monotonicity:", force_monotonicity)
    print("monotonic_range:", monotonic_range)
    print("intrinsic_guess:", intrinsic_guess if len(intrinsic_guess) else False)
    print("fix_pp:", fix_pp)
    print("threads:", threads)
    print("debug:", debug)
    print("-" * 50)

    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # prepare object points, like (0,0,0), (30,0,0), (60,0,0) ....
    # each square is 30x30mm
    # NB. the intrinsic parameters, rvec and distCoeffs do not depend upon the chessboard size, tvec does instead!
    objp = np.zeros((inner_corners_height*inner_corners_width,3), np.float32)
    objp[:,:2] = np.mgrid[0:inner_corners_height,0:inner_corners_width].T.reshape(-1,2)
    objp[:,:2] *= square_sizes

    filename_images = utils.find_images(folder_images, "*")
    if len(filename_images) == 0:
        print("!!! Unable to detect images in this folder !!!")
        sys.exit(0)

    if threads>0:
        with multiprocessing.Pool(threads) as pool:
            res = pool.starmap(process_image, zip(filename_images, repeat(inner_corners_height),
                                                  repeat(inner_corners_width), repeat(debug),
                                                  repeat(debug_folder)))
    else:
        res = [process_image(f, inner_corners_height, inner_corners_width,
                                debug, debug_folder) for f in filename_images]

    objpoints = [objp.copy() for r in res if r is not None] # 3d point in real world space
    imgpoints = [r.copy() for r in res if r is not None] # 2d points in image plane.

    image = imageio.imread(filename_images[0])
    image_shape = image.shape[:2]
    
    # visualize the keypoints
    plt.figure()
    plt.plot(*np.vstack(imgpoints).squeeze().transpose(1,0), 'g.')
    plt.grid()
    plt.xlim(0, image_shape[1])
    plt.ylim(image_shape[0], 0)
    plt.savefig(os.path.join(output_folder, "detected_keypoints.jpg"), bbox_inches='tight')
    
    calib_flag = cv2.CALIB_FIX_ASPECT_RATIO
    if rational_model:
        calib_flag += cv2.CALIB_RATIONAL_MODEL
    if fix_pp:
        calib_flag += cv2.CALIB_FIX_PRINCIPAL_POINT
        
    K_guess, dist_guess = None, None
    if len(intrinsic_guess):
        intrinsic_guess = utils.json_read(intrinsic_guess)
        K_guess = np.array(intrinsic_guess['K'])
        dist_guess = np.array(intrinsic_guess['dist'])
        calib_flag = cv2.CALIB_USE_INTRINSIC_GUESS
        
        print("K_guess:", K_guess)
        print("dist_guess:", dist_guess)

    print("working hard...")
    #ret, mtx, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_shape[::-1], None, None)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    iFixedPoint = inner_corners_height-1
    ret, mtx, distCoeffs, rvecs, tvecs, newObjPoints, \
    stdDeviationsIntrinsics, stdDeviationsExtrinsics, \
    stdDeviationsObjPoints, perViewErrors = cv2.calibrateCameraROExtended(objpoints, imgpoints, image_shape[::-1],
                                                                          iFixedPoint, K_guess, dist_guess,
                                                                          flags=calib_flag, criteria=criteria)
    
    def reprojection_error(mtx, distCoeffs, rvecs, tvecs):
        # print reprojection error
        reproj_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, distCoeffs)
            reproj_error += cv2.norm(imgpoints[i],imgpoints2,cv2.NORM_L2)/len(imgpoints2)
        reproj_error /= len(objpoints) 
        return reproj_error
    
    reproj_error = reprojection_error(mtx, distCoeffs, rvecs, tvecs)
    print("RMS Reprojection Error: {}, Total Reprojection Error: {}".format(ret, reproj_error))
    
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, distCoeffs, image_shape[::-1], alpha, 
                                                      image_shape[::-1], centerPrincipalPoint=False)
    
    grid_norm, is_monotonic = probe_monotonicity(mtx, distCoeffs, newcameramtx, image_shape, N=100, M=100)
    if not np.all(is_monotonic):
        print("-"*50)
        print(" The distortion function is not monotonous for alpha={:0.2f}!".format(alpha))
        print(" To fix this we suggest sampling more precise points on the corner of the image first.")
        print(" If this is not enought, use the option Rational Camera Model.")
        print("-"*50)
    
    # visualise monotonicity
    plt.figure()
    plt.imshow(cv2.undistort(image, mtx, distCoeffs, None, newcameramtx))
    grid = grid_norm*newcameramtx[[0,1],[0,1]][None]+newcameramtx[[0,1],[2,2]][None]
    plt.plot(grid[is_monotonic, 0], grid[is_monotonic, 1], '.g', label='monotonic', markersize=1.5)
    plt.plot(grid[~is_monotonic, 0], grid[~is_monotonic, 1], '.r', label='not monotonic', markersize=1.5)
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_folder, "monotonicity.jpg"), bbox_inches='tight')
    
    proj_undist_norm = np.vstack([cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], np.eye(3), None)[0].reshape(-1,2)
                                     for i in range(len(rvecs))])
    
    if force_monotonicity:
        is_monotonic = is_distortion_function_monotonic(distCoeffs, range=(0, monotonic_range, 1000))
        if is_monotonic:
            print("The distortion function is monotonic in the range (0,{:0.2f})".format(monotonic_range))
        else:
            print("The distortion function is not monotonic in the range (0,{:0.2f})".format(monotonic_range))

        if not is_monotonic:
            print("Trying to enforce monotonicity in the range (0,{:.2f})".format(monotonic_range))

            image_points = np.vstack(imgpoints)
            distCoeffs = enforce_monotonic_distortion(distCoeffs, mtx, image_points, proj_undist_norm, 
                                                      range_constraint=(0, monotonic_range, 1000))
            #TODO: refine mtx with the new distCoeffs in order to reduce reprojection error 

            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, distCoeffs, image_shape[::-1], alpha, 
                                                              image_shape[::-1], centerPrincipalPoint=False)
            
            rvecs_new, tvecs_new = [],[]
            for objp,imgp in zip(objpoints, imgpoints):
                _, rvec, tvec = cv2.solvePnP(objp, imgp, mtx, distCoeffs) # is this the best?
                rvecs_new.append(rvec)
                tvecs_new.append(tvec)

            reproj_error = reprojection_error(mtx, distCoeffs, rvecs_new, tvecs_new)
            print("mono: RMS Reprojection Error: {}, Total Reprojection Error: {}".format(ret, reproj_error))

    d_json = dict({"date":current_datetime, "description":description,
                   "K":mtx.tolist(), "K_new":newcameramtx.tolist(), "dist":distCoeffs.tolist(),
                   "reproj_error":reproj_error, "image_shape":image_shape})

    utils.json_write(os.path.join(output_folder, "intrinsics.json"), d_json)

    # The code from this point on as the purpose of verifiying that the estimation went well.
    # images are undistorted using the compouted intrinsics
    
    # undistorting the images
    print("Saving undistorted images..")
    for i,filenames_image in enumerate(filename_images):

        img = imageio.imread(filenames_image)
        h, w = img.shape[:2]

        try:
            dst = cv2.undistort(img, mtx, distCoeffs, None, newcameramtx)
            # to project points on this undistorted image you need the following:
            # cv2.projectPoints(objpoints, rvec, tvec, newcameramtx, None)[0].reshape(-1,2)
            # or:
            # cv2.undistortPoints(imgpoints, mtx, distCoeffs, P=newcameramtx).reshape(-1,2)
            
            # draw principal point
            dst = cv2.circle(dst, (int(mtx[0, 2]), int(mtx[1, 2])), 6, (255, 0, 0), -1)

            imageio.imsave(os.path.join(undistorted_folder, os.path.basename(filenames_image)), dst)
        except:
            print("Something went wrong while undistorting the images. The distortion coefficients are probably not good. You need to take a new set of calibration images.")
            #sys.exit(0)

if __name__ == "__main__":
    
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_images", "-i", type=str, required=True)
    parser.add_argument("--output_folder", "-o", type=str, default='./output', required=False)
    parser.add_argument("--description", "-d", type=str, default="", required=False,
                        help="Optional description to add to the output file.")
    parser.add_argument("--inner_corners_height", "-ich", type=int, required=True)
    parser.add_argument("--inner_corners_width", "-icw", type=int, required=True)
    parser.add_argument("--square_sizes", "-s", type=int, default=1, required=False)
    parser.add_argument("--alpha", "-a", type=float, default=0.95, required=False,
                        help="Parameter controlling the ammount of out-of-image pixels (\"black regions\") retained in the undistorted image.")
    parser.add_argument("--threads", "-t", type=int, default=4, required=False)
    parser.add_argument("--force_monotonicity", "-fm", type=str2bool, default=False, required=False,
                        help="Force monotonicity in the range defined by monotonic_range. To be used only in extreme cases.")
    parser.add_argument("--monotonic_range", "-mr", type=float, default=-1, required=False,
                        help="Value defining the range for the distortion must be monotonic. Typical value to try 1.3. Be careful: increasing this value may negatively perturb the distortion function.")
    parser.add_argument("--rational_model", "-rm", action="store_true", required=False,
                        help="Use a camera model that is better suited for wider lenses.")
    parser.add_argument("--fix_pp", "-p", action="store_true", required=False,
                        help="Fix the principal point either at the center of the image or as specified by intrisic guess.")
    parser.add_argument("--intrinsic_guess", "-ig", type=str, required=False, default="",
                        help="JSON file containing a initial guesses for the intrinsic matrix and distortion parameters.")
    parser.add_argument("--debug", action="store_true", required=False)
    args = parser.parse_args()
    
    main(**vars(args))  
    
#python compute_intrinsics.py --folder_images ./frames -ich 6 -icw 8 -s 30 -t 24 --debug    