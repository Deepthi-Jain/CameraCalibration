# CameraCalibration

This Python file contains definitions for the three functions:
    (A)calibrateCamera3D(data)
    (B)visualiseCameraCalibration3D(data, P)
    (C)evaluateCameraCalibration3D(data, P).
    (D)Driver function main() to call the above three functions.

(A) calibrateCamera3D(data) performs camera calibration by computing Perspective matrix P 
using eigenvalues of matrix A.

(B)visualiseCameraCalibration3D(data, P) renders a single 2D plot showing 
(i) the measured 2D image point (as given in the question), and, 
(ii) the reprojection of the 3D calibration points as computed by P.

(C)evaluateCameraCalibration3D(data, P) evaluates  mean, variance, minimum, and maximum
distances in pixels between the measured and reprojected image feature locations.


Created on Sat Nov 23 22:28:54 2019

@author: Deepthi Jain Brahmadev.
deepthinithin1920@gmail.com
