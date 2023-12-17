# 3D AprilTags autolocalization

The goal of this practice is to program a beacon-based visual autolocation system. To do this, the DroidCam application will be used to capture the video, the AprilTag library to detect the beacons, and the Matplotlib Python library to display the current estimate position in a 3D viewer.

## Calibrate the camera
The first step is to calibrate the camera to be used. To do this you need to print a checkerboard calibration template like the one in the image:

![image](https://github.com/elenagiraldo3/april_tags_autolocalization/assets/55191542/bc4cae89-493f-4d58-aa73-94c79ad9375a)

Once the template is printed, take some pictures (the more the better) of the template in different positions with the same camera that will be used for autolocation.
![image](https://github.com/elenagiraldo3/april_tags_autolocalization/assets/55191542/5267a191-0005-4b97-8e5b-b2a7436cdc21)

These images were taken with the DroidCam app. In order to use it, it is necessary to download the app both on the smartphone and computer. In order for the video to be displayed on the computer, the IP address that appears on the smartphone app must be entered in the computer app or the HTTP address in a browser.
![image](https://github.com/elenagiraldo3/april_tags_autolocalization/assets/55191542/b9f7b00a-cc57-461f-b26e-d5cdc7c4f9fa)

To perform the calibration, you must run the program `camera_calibration.py` as:

```python camera_calibration.py --folder "folder/to/images" --pshape (9,6) --psize 24.5```

where the parameter `folder` is the path to where the calibration images are located, `pshape` is the number of inner corners per a chessboard row and column and `psize` is the size of the chessboard squares in millimeters. The program will return a file `calibration.npz` containing the camera intrinsic matrix and the vector of distorsion coefficients, which will be necessary for the autolocalization.

## Testing the AprilTag library
To detect the beacons, the AprilTag library will be used. AprilTag is a visual fiducial system, useful for a wide variety of tasks including augmented reality, robotics, and camera calibration. Targets can be created from an ordinary printer, and the AprilTag detection software computes the precise 3D position, orientation, and identity of the tags relative to the camera. 

![image](https://github.com/elenagiraldo3/april_tags_autolocalization/assets/55191542/91cf6e3c-e188-4c20-abc1-451f50069097)

In particular, for this exercise the pupil-apriltags library has been used, which must be taken into account that only runs on **Python 3.6 or 3.7**. In this video you can see the detection of a beacon with this library:

[AprilTag Detection](https://www.youtube.com/watch?v=QKN2ABrs0U0)

## Estimate actual position

The estimation of the current position is done in the program `detect_apriltag.py`.

There are a number of parameters that need to be changed to match the scenario being tested:

- In line 49, the name of the NPZ must be indicated.
- In line 50, the size of the tag squares in millimeters must be indicated.
- In line 51, the family of the tags must be indicated.
- In line 52, the number of the camera that will be use must be indicated.
- In line 53, the tag's id that will be use must be indicated.
- In line 54 it must be indicated the coordinates of the 4 corners of each tag in the world coordinate system (in millimeters). They must be in the same orders as the tag's id.

![image](https://github.com/elenagiraldo3/april_tags_autolocalization/assets/55191542/a3154fcd-f1a9-4213-91cb-22b0596115c4)

The steps followed in this program are:

1. Load the camera intrinsic matrix and the vector of distorsion coefficients from the NPZ file.
2. Capture the video that is being received from the DroidCam.
3. Initialize apriltags detector.
4. Create the figure where the autolocation will be shown with the help of the matplotlib library and plot squares that represent the tags.
5. Next, for each frame, the image is converted to grayscale and the apriltags detector is used to detect all the tags that appear in the image. For each detected tag, a bounding box and the tag id are drawn in an opencv window.
6. For each detected tag, the detector provides us with the coordinates of the corners in the image coordinate system. Knowing the coordinates in the image and their correspondence in the world, we can obtain the position of the camera using the solvePnP function. This function returns the rotation and the translation vectors that transform a 3D point expressed in the object coordinate system to the camera coordinate system. 
7. We use the Rodrigues function to get the rotation matrix (solvePnP returns a vector).
8. We calculate the optical center of the camera using the equation: $C = -R.T @ t$ where R.T is the transpose matrix of the rotation matrix and t is the translation vector.
9. More than one tag may appear in an image so it is useful to fuse all the individual estimations from each marker, hopefully improving the robustness of the final estimation. The 3D fusion performed is a weighted average of the coordinates and angles of all the estimated poses: the closer the tag is, the bigger is the weight assigned. This fusion is done in every received image, so an estimation of the absolute pose of the camera is continuously available.
10. The weight of each tag is given by its area in the image. To get the mean coordinates and the mean angles, the following equations are used:
    
$$ ratio_i = \frac{weight_i}{weight_{total}} $$

$$ [x, y, z]_{fusion} = \sum ([x_i, y_i, z_i] \cdot ratio_i) $$

$$ \alpha_{fusion} = atan \left( \frac{\sum (\sin(\alpha_i) \cdot ratio_i)}{\sum (\cos(\alpha_i) \cdot ratio_i)} \right) $$

12. Finally, the current position of the camera is painted in the matplotlib viewer. A string of the last 15 camera positions is also displayed.

Below, you can see a video of the autolocation achieved using three tags in different positions and rotations:

[3D Autolocalization using AprilTags](https://www.youtube.com/watch?v=afpQyWUJaz0)
