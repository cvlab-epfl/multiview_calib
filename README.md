# Multiple view Camera calibration tool
This tool allows to compute the intrinsic and extrinsic camera parameters of a set of synchronized cameras with overlapping field of view. The intrinsics estimation is based on the OpenCV's camera calibration framework and it is used on each camera separately. In the extrinsics estimation, an initial solution (extrinsic parameters) is computed first using a linear approach then refined using bundle adjustment.  The output are camera poses (intrinsic matrix, distortion parameters, rotations and translations) w.r.t. either the first camera or a global reference system.

## Prerequisites

- numpy
- scipy
- imageio
- matplotlib
- OpenCV

## Installation
```
cd MULTIVIEW_CALIB_MASTER
pip install .
```

## Usage

## Intrinsics estimation
#### Compute intrinsic parameters:
Print the following checkerboard and make sure the rectangles are 3x3cm. If they are not make sure to remove any configuration of the printer i.e. autofit
https://markhedleyjones.com/storage/checkerboards/Checkerboard-A4-30mm-8x6.pdf

The inner corner of the checkerboard are the calibration points.

Take a video of the checkerboard. The objective is to acquire a set of images (30-200, 2min video) of the checkerboard from different viewpoints by making sure that the distribution of the calibration points covers the whole image, corner comprises!

Extract the frames:
```
ffmpeg -i VIDEO -r 0.5 frames/frame_%04d.jpg
```
Run the following script:
```
python compute_intrinsics.py --folder_images ./frames -ich 6 -icw 8 -s 30 -t 24 --debug
```
The script outputs several useful information for debugging purposes. One of them is the per keypoint reprojection error, another the monotonicity of the distortion function. If the distortion function is not monotonic, we suggest to sample more precise points on the corner of the image first. If this is not enought, try the Rational Model (-rm) instead. The Rational Model is a model of the lens that is more adapted to cameras with wider lenses.
To furter understand if the calibration went well, you should perform a visual inspection of the undistorted images that have been saved. The lines in the images should be straight and the picture must look like a normal picture. In case of failure try to update Opencv or re-take the video/pictures.

## Extrinsics estimation
#### Synchronization:
In the case the phyiscal landmarks that you want to use to calibrate the camera are not static, you have to synchronize the cameras. We do this by extracting the frames from each one of the videos using the exact same frame rate (the higher the better), then, we look for a fast and recognizible event in the videos (like a hand clap) that allow us to remove the time offset in term of frame indexes from the sequences. Once the offset is removed you can then locate the landmarks in each sequence.

To extract the frames:
```
ffmpeg -i VIDEO -vf "fps=30" frames/frame_%06d.jpg
```
It is a good idea to extarct the frame arround the native frame rate. Increasing the fps w.r.t the original fps would not make the synchornization more precise.

#### Compute relative poses:
To recover the pose of each one of the cameras in the rig w.r.t. the first camera we first compute relative poses between pairs of views and then concatenate them to form a tree. To do so, we have to manually define a minimal set of pairs of views that connect every camera. This is done in the file `setup.json`.
```diff
-Note: do not pair cameras that are facing eachother! Recovering proper geometry in this specifc case is difficult.
```
The file named `landmarks.json` contains precise image points for each view that are used to compute fundamental matrices and poses. The file `ìntrinsics.json` contains the intrinsic parameters for each view that we have computed previously. The file `filenames.json` contains a filename of an image for each view which are is used for visualisation purposes.
Check section `Input files` for more details on the file formats.

```
python compute_relative_poses.py -s setup.json -i intrinsics.json -l landmarks.json -f filenames.json --dump_images 
```
The result of this operation are relative poses up to scale (the translation vector is unit vector).

The following command is an alternative. It computes the final relative pose from view1 to view2 as an average of relative poses computed using N other and different views.
```
python compute_relative_poses_robust.py -s setup.json -i intrinsics.json -l landmarks.json -m lmeds -n 5 -f filenames.json --dump_images
```

#### Concatenate relative poses:
In this step we concatenate/chain all the relative poses to obtain an approximation of the actual camera poses. The poses are defined w.r.t the first camera. At every concatenation we scale the current relative pose to match the scale of the previous ones. This to have roughly the same scale for each camera.
The file `relative_poses.json` is the output of the previous step.
```
python concatenate_relative_poses.py -s setup.json -r relative_poses.json --dump_images 
```
#### Bundle adjustment:
Nonlinear Least squares refinement of intrinsic and extrinsic parameters and 3D points. The camera poses output of this step are up to scale.
The file `poses.json` is the output of the previous step (Concatenate relative poses).
```
python bundle_adjustment.py -s setup.json -i intrinsics.json -e poses.json -l landmarks.json -f filenames.json --dump_images -c ba_config.json 
```
#### Transformation to the global reference system:
The poses and 3D points computed using the bundle adjustment are all w.r.t. the first camera and up to scale.
In order to have the poses in the global/world reference system we have to estimate the rigid transformation between the two reference systems. To do so we perform a rigid allignement of the 3D points computed using bundle adjustment and their corresponding ones in global/world coordinate (at least 4 non-symmetric points). These must be defined in the file `landmarks_global.json` and have the same ID of the points defined in `landmarks.json`. Note that there is no need to specify the global coordinate for all landmarks defined in `landmarks.json`; a subset is enough. Given these correspondeces, the following command will find the best rigid transform in the least squares sense between the two point sets and then update the poses computed by the bundle adjustment. The output are the update poses saved in `global_poses.json`. NOTE: make sure the points used here are not symmetric nor close to be symmetric as this implies multiple solutions whcih is not handeled!
```
python global_registration.py -s setup.json -ps ba_poses.json -po ba_points.json -l landmarks.json -lg landmarks_global.json -f filenames.json --dump_images  
```
If the global landmarks are a different set of points than the one used during the optimization, you can use the following command to compute the `ba_points.json`.
```
python triangulate_image_points.py -p ba_poses.json -l landmarks.json --dump_images
```
## Input files
The file `setup.json` contains the name of the views and the minimal number of pairs of views that allows to connect all the cameras togheter. `minimal_tree` is a tree and is single component, therefore, it cannot for loops and all views are connected.
```json
{
 "views": [ "cam0", "cam1", "cam2", "cam3"], 
 "minimal_tree": [["cam0","cam1"], ["cam1","cam2"], ["cam3","cam0"]]
}
```
The file `landmarks.json` contains the image points use to compute the poses. An image point is the projection of a landmark that exist in the physcal space. A unique ID is associated to each landmark. If the same landmark is visible in other views the same ID should be used. If the landmark is a moving object, make sure your cameras are synchronized and that you assign a different ID from frame to frame. Have a look at the examples if this is not clear enough. 
```json
{
 "cam0":{"landmarks": [[530.1256, 877.56], [2145.5564, 987.4574], ..., [1023, 126]],
         "ids": [0, 1, ..., 3040]},
 ...
 "cam3":{"landmarks": [[430.1256, 377.56], [2245.5564, 387.4574], ..., [2223, 1726]], 
         "ids": [1, 2, ..., 3040]}         
}
```
The file `landmarks_global.json` contains 3D points defined in the "global" reference system. These defines the global location of all or a subset of the landmarks in the file `landmarks.json`. The IDs in this file must therefore allign with the IDs in the file `landmarks.json` but is not required that all landmarks have a global coordinate. The global points can be GPS coordinates in UTM+Altitude format or simply positions w.r.t. any other reference the you want. The global point can be noisy.
```json
{
 "landmarks_global": [[414278.16, 5316285.59, 5], [414278.16, 5316285.59, 5.5], ..., [414278.16, 5316285.59, 5.2]],
 "ids": [0, 1, ..., 3040]}       
}
```
The file `intrinsics.json` contains the instrinsics parameters in the following format:
```json
{
 "cam0": { "K": [[1798.760123221333, 0.0, 1947.1889719803005], 
                  [0.0, 1790.0624403935456, 1091.2910152343356],
                  [ 0.0, 0.0, 1.0]],
            "dist": [-0.22790810,0.0574260,0.00032600,-0.00047905,-0.0068488]},
 ...           
 "cam3": { "K": [[1778.560123221333, 0.0, 1887.1889719803005], 
                  [0.0, 1780.0624403935456, 1081.2910152343356],
                  [ 0.0, 0.0, 1.0]],
            "dist": [-0.2390810,0.0554260,0.00031600,-0.00041905,-0.0062488]}
}
```
The file `filenames.json` contains one filename for each view. It is used for visualisation purposes only:
```json
{
 "cam0": "somewhere/filename_cam0.jpg",
 ...           
 "cam3": "somewhere/filename_cam3.jpg",
}
```
The file `ba_config.json` contains the configuration for the bundle adjustment. A typical configuration is the following:
```json
{
  "each_training": 1
  "each_visualisation": 1,
  "th_outliers_early": 1000.0,
  "th_outliers": 50,
  "optimize_points": true,
  "optimize_camera_params": true,
  "bounds": true,  
  "bounds_cp": [ 
    0.3, 0.3, 0.3,
    2, 2, 2,
    10, 10, 10, 10,
    0.01, 0.01, 0, 0, 0
  ],
  "bounds_pt": [
    1000,
    1000,
    1000
  ],
  "max_nfev": 200,
  "max_nfev2": 200,
  "ftol": 1e-08,
  "xtol": 1e-08,  
  "loss": "linear",
  "f_scale": 1,
  "output_path": "output/bundle_adjustment/",
}
```
## License

