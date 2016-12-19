"""Toolbox for advanced lane finding!
"""

import os
import sys

import numpy as np
import cv2

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# ============================================================================ #
# Plotting methods.
# ============================================================================ #
def plot_dual(img1, img2, title1='', title2='', figsize=(24, 9)):

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img1)
    ax1.set_title(title1, fontsize=25)
    ax2.imshow(img2)
    ax2.set_title(title2, fontsize=25)
    # f.show()


# ============================================================================ #
# Calibration methods.
# ============================================================================ #
def calibration_parameters(path, cshape):
    """Compute calibration parameters from a set of calibration images.

    Params:
      path: Directory of calibration images.
      cshape: Shape of grid used in the latter.
    Return:
      mtx, dist
    """
    # Object / image points collections.
    objpoints = []
    imgpoints = []

    # Calibration points from images.
    filenames = os.listdir(path)
    for fname in filenames:
        img = cv2.imread(path + fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Theoretical Grid.
        objp = np.zeros((cshape[0] * cshape[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:cshape[0], 0:cshape[1]].T.reshape(-1, 2)
        # Corners in the image.
        ret, corners = cv2.findChessboardCorners(gray, cshape, None)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
        else:
            print('Warning! Not chessboard found in image', fname)

    # Calibration from image points.
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[0:2], None, None)
    return mtx, dist


def test_calibration(fname, cshape, mtx, dist):
    """Test calibration on an example chessboard, and display the result.
    """
    # Load image, draw chessboard and undistort.
    img = cv2.imread(fname)
    ret, corners = cv2.findChessboardCorners(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                                             cshape, None)
    img = cv2.drawChessboardCorners(img, cshape, corners, ret)
    undst = cv2.undistort(img, mtx, dist, None, mtx)

    # Plot results.
    plot_dual(img, undst,
              title1='Original Chessboard',
              title2='Undistorted Chessboard', figsize=(24, 9))
