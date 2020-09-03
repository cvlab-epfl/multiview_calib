# Multiple view calibration tool

## Prerequisites

- numpy
- scipy
- imageio
- matplotlib
- OpenCV

## Installation
```
export PYTHONPATH="...parent folder...:$PYTHONPATH"
```

## Usage

#### Compute intrinsic parameters:
Print the following checkerboard and make sure the rectangles are 3x3cm. If they are not make sure to remove any configuration of the printer i.e. autofit
https://markhedleyjones.com/storage/checkerboards/Checkerboard-A4-30mm-8x6.pdf

Take a video of the checkerboard. The camera is fixed in the same position whereas the checkerboard moves in front of the camera. The checkerboard should cover the whole image plane. The inner corners of the peripheral squares should be as close as possible to the border of the image. A 2min video is usually enough.

Extract the frames:
```
ffmpeg -i VIDEO -r 0.5 frames/frame_%04d.jpg
```
Run the following script:
```
python compute_intrinsics.py --folder_images ./frames -ich 6 -icw 8 -s 30 -t 24 --debug
```
To understand if the calibration is correct you have to perform a visual inspction of the undistorted images that have been saved. The lines in the images should be straight and the picture has to look like a normal picture. In case of failure try to update Opencv or re-take the video/pictures.

#### Compute relative poses:
At this step we compute the relative poses between pairs of views. These rigid transformations will then be chained one another to compute the pose of each camera w.r.t the first camera. In order to do so, we have to manually define a minimal set of pairs of views that connect every camera. This is done in the file `setup.json`. 
The file named `landmarks.json` contains precise image points for each view that are used to compute fundamental matrices and poses. The file `ìntrinsics.json` contains the intrinsic parameters for each view. The file `filenames.json` contains one filename of an image for each view; it is used for visualisation purposes.
Check section `Input files` for more details on the file formats.

```
python compute_relative_poses.py -s setup.json -i intrinsics.json -l landmarks.json -f filenames.json --dump_images 
```
The result of this operation are relative poses from the first to second camera for each pairs of views. The poses are up to scale, in other words the translation is a unit vector.

The following command is an alternative. It computes the final relative pose from view1 to view2 as an average of relative poses computed using N other and different views.
```
python compute_relative_poses_robust.py -s setup.json -i intrinsics.json -l landmarks.json -m lmeds -n 5 -f filenames.json --dump_images
```

#### Concatenate relative poses:
In this step we concatenate/chain all the relative poses to obtain an estimate of the actual camera poses. The poses are defined w.r.t the first camera. At every concatenation we scale the current relative pose to match the scale of the previous ones. This to have roughly the same scale for each camera.
The file `relative_poses.json` is the output of the previous step.
```
python concatenate_relative_poses.py -s setup.json -r relative_poses.json --dump_images 
```
#### Bundle adjustment:
Least squares refinement of intrinsic and extrinsic parameters and 3D points. The camera rig is still up to a scale at this point.
The file `poses.json` is the output of the previous step.
```
python bundle_adjustment.py -s setup.json -i intrinsics.json -e poses.json -l landmarks.json -f filenames.json --dump_images -c ba_config.json 
```
#### Transformation to the global reference system:
The poses and 3D points computed using the bundle adjustment are all w.r.t the reference system provided by the first camera. To transform the poses into the world coordinate system we require correspondences between the landmarks in the image and the world. These must be defined in the file `landmarks_global.json`. Given these correspondeces, the following command will find the best rigid transform in the least squares sense between the two point sets and then update the poses computed with the bundle adjustment. The global coordinates can be noisy. The output are the update poses saved in `global_poses.json`.
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
The file `landmarks.json` contains the image points use to compute the poses. An image point is the projection of a landmark that exist in the physcal space. A unique ID is associated to each landmark. If the same landmark is visible in other views the same ID is used. A landmark may not be visible in all views. 
```json
{
 "cam0":{"landmarks": [[530.1256, 877.56], [2145.5564, 987.4574], ..., [1023, 126]],
         "ids": [0, 1, ..., 3040]},
 ...
 "cam3":{"landmarks": [[430.1256, 377.56], [2245.5564, 387.4574], ..., [2223, 1726]], 
         "ids": [1, 2, ..., 3040]}         
}
```
The file `landmarks_global.json` contains 3D points defined in the "global" reference system. These defines the global location of the landmarks in the file `landmarks.json`. The IDs in this file must therefore allign with the IDs in the file `landmarks.json` but is not required that all landmarks have a global coordinate. The global points can be GPS coordinates in UTM+Altitude format or simply positions w.r.t any other reference the you want. The global point can be noisy.
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

