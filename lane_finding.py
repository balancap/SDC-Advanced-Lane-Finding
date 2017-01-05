"""Toolbox for advanced lane finding!
"""

import os
import sys
import json

import numpy as np
import cv2

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# =========================================================================== #
# Plotting and drawing methods.
# =========================================================================== #
def plot_images(imgs, titles, cmap='gray', figsize=(24, 9)):
    nimgs = len(imgs)
    f, axes = plt.subplots(1, nimgs, figsize=figsize)
    f.tight_layout()
    for i in range(nimgs):
        axes[i].imshow(imgs[i], cmap=cmap)
        axes[i].set_title(titles[i], fontsize=25)


def plot_dual(img1, img2, title1='', title2='', figsize=(24, 9)):

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    f.tight_layout()
    ax1.imshow(img1)
    ax1.set_title(title1, fontsize=25)
    ax2.imshow(img2)
    ax2.set_title(title2, fontsize=25)
    # f.show()


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """Draw a collection of lines on an image.
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def draw_mask(img, mask, color=[255, 0, 0], alpha=0.8, beta=1., gamma=0.):
    """The result image is computed as follows: img * α + mask * β + λ
    where mask is transformed into an RGB image using color input.
    """
    mask = (mask > 0).astype(np.uint8)
    color_mask = np.dstack((color[0] * mask, color[1] * mask, color[2] * mask))
    return cv2.addWeighted(img, alpha, color_mask, beta, gamma)


# =========================================================================== #
# Calibration methods.
# =========================================================================== #
def undistort_image(img, mtx, dist):
    return cv2.undistort(img, mtx, dist, None, mtx)


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
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,
                                                       imgpoints,
                                                       img.shape[0:2],
                                                       None, None)
    return mtx, dist


def test_calibration(fname, cshape, mtx, dist):
    """Test calibration on an example chessboard, and display the result.
    """
    # Load image, draw chessboard and undistort.
    img = cv2.imread(fname)
    ret, corners = cv2.findChessboardCorners(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                                             cshape, None)
    img = cv2.drawChessboardCorners(img, cshape, corners, ret)
    undst = undistort_image(img, mtx, dist)

    # Plot results.
    plot_dual(img, undst,
              title1='Original Chessboard',
              title2='Undistorted Chessboard', figsize=(24, 9))


# ============================================================================ #
# Perspective transform.
# ============================================================================ #
def apply_scaling(x, y, scale, shape, reversed_x=True, dtype=np.float32):
    """Apply scaling factor to X and Y vectors.
    """
    if reversed_x:
        x = shape[0] - 1 - x
    x = x / scale[0]
    y = y / scale[1]
    return x.astype(dtype), y.astype(dtype)


def warp_image(img, mtx_perp, flags=cv2.INTER_LINEAR):
    img_size = (img.shape[1], img.shape[0])
    return cv2.warpPerspective(img, mtx_perp, img_size, flags=cv2.INTER_LINEAR)


def perspective_transform(src_points, dst_points):
    """Compute perspective transform from source and destination points.
    """
    mtx_perp = cv2.getPerspectiveTransform(src_points, dst_points)
    mtx_perp_inv = cv2.getPerspectiveTransform(dst_points, src_points)
    return mtx_perp, mtx_perp_inv


def test_perspective(img, src_points, mtx_perp):
    """Test the perspective transform on a sample image.
    """
    # Ugly hack to print lines!
    l = len(src_points)
    lines = [[[src_points[i][0],
               src_points[i][1],
               src_points[(i+1) % l][0],
               src_points[(i+1) % l][1]]] for i in range(l)]
    draw_lines(img, lines, thickness=2)

    # Apply transform.
    wimg = warp_image(img, mtx_perp, flags=cv2.INTER_LINEAR)
    # unwimg = cv2.warpPerspective(wimg, m_inv_perp, img_size, flags=cv2.INTER_LINEAR)
    # Plot result.
    plot_dual(img, wimg,
              title1='Original image.',
              title2='Warped image.', figsize=(24, 9))


# =========================================================================== #
# Loading / Saving methods
# =========================================================================== #
def load_image(filename, crop_shape=(720, 1280)):
    """Load an image, and crop it to the correct shape if necessary.
    """
    img = mpimg.imread(filename)
    shape = img.shape
    # Cropping.
    if shape[0] > crop_shape[0]:
        img = img[-crop_shape[0]:, :, :]
    if shape[1] > crop_shape[1]:
        img = img[:, :crop_shape[1], :]
    return img


def load_points(filename, key, dtype=np.float32):
    """Load data points from a json file.
    """
    with open(filename) as fdata:
        jdata = json.load(fdata)
    data = np.array(jdata.get(key, []), dtype=dtype)
    return data


# ============================================================================ #
# Points / Lanes / Masks...
# ============================================================================ #
def masks_to_points(wmasks, add_mask, order=2,
                    reverse_x=True, normalise=True, dtype=np.float32):
    """Construct the collection of points from masks.
    """
    shape = add_mask.shape
    x = np.zeros((0,))
    y = np.zeros((0,))
    for wm in wmasks:
        # Left points.
        x0, y0 = np.where(wm * add_mask)
        x = np.append(x, x0)
        y = np.append(y, y0)
    # Reverse X axis: zero to bottom of image.
    if reverse_x:
        x = shape[0] - x - 1
    if normalise:
        x = x.astype(dtype) / shape[0]
        y = y.astype(dtype) / shape[1] - 0.5

    # Construct big vector! Assume order-2 model.
    X = np.zeros((len(x), order+1), dtype=dtype)
    X[:, 0] = 1.
    for i in range(1, order+1):
        X[:, i] = x**i
    return X.astype(dtype), y.astype(dtype)


def predict_lanes(model_lanes, wimg, order=2,
                  reversed_x=True, normalised=True, dtype=np.float32):
    shape = wimg.shape
    x = np.arange(0, shape[0]).astype(dtype)
    # Normalise x values.
    if reversed_x:
        x = shape[0] - x - 1
    if normalised:
        x = x / shape[0]

    # Prediction.
    X = np.zeros((len(x), order+1), dtype=dtype)
    X[:, 0] = 1.
    for i in range(1, order+1):
        X[:, i] = x**i
    y1, y2 = model_lanes.predict(X, X)

    # De-normalise!
    if normalised:
        x = (x) * shape[0]
        X = np.vstack((np.ones(shape[0], ), x, x**2)).T
        y1 = (0.5 + y1) * shape[1]
        y2 = (0.5 + y2) * shape[1]
    if reversed_x:
        x = shape[0] - x - 1
        X = np.vstack((np.ones(shape[0], ), x, x**2)).T

    return X, y1, y2


def predict_lanes_w(w1, w2, wimg, order=2,
                    reversed_x=True, normalised=True, dtype=np.float32):
    shape = wimg.shape
    x = np.arange(0, shape[0]).astype(dtype)
    # Normalise x values.
    if reversed_x:
        x = shape[0] - x - 1
    if normalised:
        x = x / shape[0]

    # Prediction.
    X = np.zeros((len(x), order+1), dtype=dtype)
    X[:, 0] = 1.
    for i in range(1, order+1):
        X[:, i] = x**i
    y1 = X @ w1
    y2 = X @ w2

    # De-normalise!
    if normalised:
        x = (x) * shape[0]
        X = np.vstack((np.ones(shape[0], ), x, x**2)).T
        y1 = (0.5 + y1) * shape[1]
        y2 = (0.5 + y2) * shape[1]
    if reversed_x:
        x = shape[0] - x - 1
        X = np.vstack((np.ones(shape[0], ), x, x**2)).T

    return X, y1, y2


def lanes_to_wmask(wimg, x1, y1, x2, y2):
    """Generate the lane mask using left and right lanes.
    """
    shape = wimg.shape[0:2]

    # Create an image to draw the lines on
    warp_zero = np.zeros(shape, np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([y1, x1]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([y2, x2])))])
    pts = np.hstack((pts_left, pts_right))
    pts = np.array([pts], dtype=np.int64)

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, pts, (0, 255, 0))
    return color_warp


def rescale_coefficients(wimg, w, scaling, normalised=True):
    """Rescale regression coefficient using a given scaling.
    """
    shape = wimg.shape
    if normalised:
        scaling = [scaling[0] * shape[0], scaling[1] * shape[1]]
    # Re-scale coefficients.
    w_scaled = np.copy(w)
    w_scaled[0] = w[0] * scaling[1]
    w_scaled[1] = w[1] * scaling[1] / scaling[0]
    w_scaled[2] = w[2] * scaling[1] / scaling[0]**2
    return w_scaled


def lane_curvature(w):
    """Compute curvature from regression coefficients.
    """
    if w[2] != 0.:
        curv = (1 + w[1]**2)**1.5 / (2*w[2])
    else:
        curv = np.inf
    return curv
