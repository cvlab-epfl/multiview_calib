import numpy as np
import cv2
import imageio

from multiview_calib import utils

inner_corners_height = 6
inner_corners_width = 8

intrinsics = utils.json_read("output/intrinsics.json")
K = np.float32(intrinsics['K'])
K_new = np.float32(intrinsics['K_new'])
dist = np.float32(intrinsics['dist'])

img = imageio.imread("frames/frame_0027.png")
gray = img[:,:,0]

#-----------------
# get camera pose
#-----------------
ret, corners = cv2.findChessboardCorners(gray, (inner_corners_height,inner_corners_width),
                                         cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
image_points = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

object_points = np.zeros((inner_corners_height*inner_corners_width,3), np.float32)
object_points[:,:2] = np.mgrid[0:inner_corners_height,0:inner_corners_width].T.reshape(-1,2)

retval, rvec, tvec = cv2.solvePnP( object_points, image_points, K, dist)

#-----------------
# undistort image
#-----------------
undist = cv2.undistort(img, K, dist, None, None)
undist_all = cv2.undistort(img, K, dist, None, K_new)

#-----------------
# project points
#-----------------
proj_obj = cv2.projectPoints(object_points, rvec, tvec, K, dist)[0].reshape(-1,2)
proj_obj_undist = cv2.projectPoints(object_points, rvec, tvec, K, None)[0].reshape(-1,2)
proj_obj_undist_all = cv2.projectPoints(object_points, rvec, tvec, K_new, None)[0].reshape(-1,2)

#-----------------
# visualisation
#-----------------

for (x,y) in proj_obj:
    img = cv2.circle(img, (int(x),int(y)), 4, [255,0,0], thickness=-1, lineType=-1)
    
for (x,y) in proj_obj_undist:
    undist = cv2.circle(undist, (int(x),int(y)), 4, [255,0,0], thickness=-1, lineType=-1) 
    
for (x,y) in proj_obj_undist_all:
    undist_all = cv2.circle(undist_all, (int(x),int(y)), 4, [255,0,0], thickness=-1, lineType=-1)     
    
imageio.imwrite("example.jpg", np.hstack([img, undist, undist_all]))