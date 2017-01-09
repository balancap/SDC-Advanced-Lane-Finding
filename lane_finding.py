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
    """Plot two images side by side.
    """
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
    """Undistort an image.
    """
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
    """Warp an image using a transform matrix.
    """
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


# =========================================================================== #
# Gradients and lanes masks computations.
# =========================================================================== #
def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def gradient_magnitude(gray, sobel_kernel=3):
    """Compute mask based on gradient magnitude. Input image assumed
    to be two dimensional.
    """
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Norm and rescaling (c.f. different kernel sizes)
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(gradmag) / 255.
    gradmag = (gradmag / scale_factor)
    return gradmag


def gradient_x(gray, sobel_kernel=3):
    """Compute mask based on horizontal gradient. Input image assumed
    to be two dimensional.
    """
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    gradmag = sobelx
    scale_factor = np.max(gradmag) / 255.
    gradmag = (gradmag / scale_factor)
    return gradmag


def gradient_y(gray, sobel_kernel=3):
    """Compute mask based on horizontal gradient. Input image assumed
    to be two dimensional.
    """
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = sobely
    scale_factor = np.max(gradmag) / 255.
    gradmag = (gradmag / scale_factor)
    return gradmag


def mask_local_crossing_x(gray, threshold=20, dilate_kernel=(2, 6), iterations=3):
    # Increasing mask.
    mask_neg = (gray < -threshold).astype(np.float32)
    mask_pos = (gray > threshold).astype(np.float32)

    mid = dilate_kernel[1] // 2
    # Dilate mask to the left.
    kernel = np.ones(dilate_kernel, np.uint8)
    kernel[:, 0:mid] = 0
    dmask_neg = cv2.dilate(mask_neg, kernel, iterations=iterations) > 0.
    # Dilate mask to the right.
    kernel = np.ones(dilate_kernel, np.uint8)
    kernel[:, mid:] = 0
    dmask_pos = cv2.dilate(mask_pos, kernel, iterations=iterations) > 0.
    dmask = (dmask_pos * dmask_neg).astype(np.uint8)

    # Eroding a bit
    # kernel = np.ones((1,2),np.uint8)
    # dmask = cv2.erode(dmask, kernel, iterations=5)
    return dmask


def mask_threshold(gray, threshold=(0, 255)):
    mask = np.zeros_like(gray)
    mask[(gray >= threshold[0]) & (gray <= threshold[1])] = 1
    return mask


def color_threshold(gray, threshold=(0, 255)):
    mask = np.zeros_like(gray)
    mask[(gray >= threshold[0]) & (gray <= threshold[1])] = 1
    return mask


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def warped_masks_original(img, m_perp, thresholds=[20, 25]):
    """Generate a collection of masks useful to detect lines.
    Original definition, calculating gradients and then apply transform.
    """
    wmasks = []
    # Grayscale and HSL images
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hsl = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

#     gammas = [0.3, 1., 3.]
    gammas = [1.]
    for g in gammas:
        # img_gamma = adjust_gamma(img, gamma=g)

        # Compute gradients.
        skernel = 13
        sobel_dx = gradient_x(gray, sobel_kernel=skernel)
        s_sobel_dx = gradient_x(hsl[:, :, 2], sobel_kernel=skernel)

        # Warped gradients.
        wsobel_dx = warp_image(sobel_dx, m_perp, flags=cv2.INTER_LANCZOS4)
        ws_sobel_dx = warp_image(s_sobel_dx, m_perp, flags=cv2.INTER_LANCZOS4)

        # Try to detect gradients configuration corresponding to lanes.
        mask = mask_local_crossing_x(wsobel_dx, threshold=thresholds[0],
                                     dilate_kernel=(2, 8), iterations=3)
        wmasks.append(mask)
        mask = mask_local_crossing_x(ws_sobel_dx, threshold=thresholds[1],
                                     dilate_kernel=(2, 8), iterations=3)
        wmasks.append(mask)

    return wmasks


def warped_masks(img, m_perp, thresholds=[20, 25]):
    """Generate a collection of masks useful to detect lines.
    """
    wmasks = []
    # Grayscale and HSL images
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hsl = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

#     gammas = [0.3, 1., 3.]
    gammas = [1.]
    for g in gammas:
        img_gamma = adjust_gamma(img, gamma=g)
        # Warped images.
        wimg = warp_image(img_gamma, m_perp, flags=cv2.INTER_LANCZOS4)
        wgray = cv2.cvtColor(wimg, cv2.COLOR_RGB2GRAY)
        whsl = cv2.cvtColor(wimg, cv2.COLOR_RGB2HLS)

        # Compute gradients.
        skernel = 13
        wsobel_dx = gradient_x(wgray, sobel_kernel=skernel)
        ws_sobel_dx = gradient_x(whsl[:, :, 2], sobel_kernel=skernel)

        # Try to detect gradients configuration corresponding to lanes.
        mask = mask_local_crossing_x(wsobel_dx, threshold=thresholds[0],
                                     dilate_kernel=(2, 8), iterations=3)
        wmasks.append(mask)
        mask = mask_local_crossing_x(ws_sobel_dx, threshold=thresholds[1],
                                     dilate_kernel=(2, 8), iterations=3)
        wmasks.append(mask)

    return wmasks


def default_left_right_masks(img, margin=0.1):
    """Default left and right masks used to find lanes: middle split with some additional margin.
    """
    shape = img.shape[0:2]
    llimit = int(shape[1] / 2 + shape[1] * margin)
    rlimit = int(shape[1] / 2 - shape[1] * margin)

    # Mask from meshgrid.
    xv, yv = np.mgrid[0:shape[0], 0:shape[1]]
    lmask = yv <= llimit
    rmask = yv >= rlimit

    return lmask, rmask


# ============================================================================ #
# Points / Lanes / Masks transforms.
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
    """Predict lanes using regression coefficients.
    """
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
    """Predict lanes using regression coefficients.
    """
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


class LanesFit():
    """Lanes fit class: contain information on a previous fit of lanes.
    """
    def __init__(self):
        # Interpolation coefficients.
        self.w_left = np.zeros((3,), dtype=np.float32)
        self.w_right = np.zeros((3,), dtype=np.float32)
        # Fitting score.
        self.fit_score_left = 0.0
        self.fit_score_right = 0.0

        # Scaling and original shape.
        self.shape = (1, 1)
        self.scaling = (1., 1.)
        self.shift = (0., 0.5)
        self.reversed = True

        # Radius of curvature, in w units.
        self.radius = 0.0
        self.line_base_pos = 0.0

    def init_from_regressor(self, regr):
        """Initialise values from a regression object.
        """
        self.w_left = regr.w1_
        self.w_right = regr.w2_
        self.fit_score_left = float(np.sum(regr.inlier_mask1_)) / regr.inlier_mask1_.size
        self.fit_score_right = float(np.sum(regr.inlier_mask2_)) / regr.inlier_mask2_.size

    def translate_coefficients(self, delta):
        """Translate lanes while keeping same curvature center.
        """
        w_left = self.w_left
        w1 = np.copy(self.w_left)
        w1[0] = delta + w_left[0]
        w1[1] = w_left[1]
        w1[2] = w_left[2] * (1 + w_left[1]**2)**1.5 / ((1 + w_left[1]**2)**1.5 - 2*delta*w_left[2])

        w_right = self.w_right
        w2 = np.copy(self.w_right)
        w2[0] = delta + w_right[0]
        w2[1] = w_right[1]
        w2[2] = w_right[2] * (1 + w_right[1]**2)**1.5 / ((1 + w_right[1]**2)**1.5 - 2*delta*w_right[2])
        return w1, w2

    def masks(self, delta):
        """Compute lanes mask, using a +- delta on every lane.
        """
        delta = np.abs(delta)
        # Meshgrid
        xv, yv = np.mgrid[0:self.shape[0], 0:self.shape[1]]
        xv = xv / float(self.scaling[0]) - self.shift[0]
        yv = yv / float(self.scaling[1]) - self.shift[1]
        if self.reversed:
            xv = xv[::-1]

        # Left part of the masks.
        w1, w2 = self.translate_coefficients(delta)
        y1 = w1[0] + w1[1] * xv + w1[2] * xv**2
        y2 = w2[0] + w2[1] * xv + w2[2] * xv**2
        lmask = yv <= y1
        rmask = yv <= y2

        # Right part of the masks.
        w1, w2 = self.translate_coefficients(-delta)
        y1 = w1[0] + w1[1] * xv + w1[2] * xv**2
        y2 = w2[0] + w2[1] * xv + w2[2] * xv**2
        lmask = np.logical_and(lmask, yv >= y1)
        rmask = np.logical_and(rmask, yv >= y2)

        return lmask, rmask


def lane_mask(shape, x, y, width):
    mask = np.zeros(shape, dtype=np.uint8)
    for w in range(width):
        xx = np.maximum(np.minimum(x, shape[0]-1), 0).astype(np.int)
        yy = np.maximum(np.minimum(y+w, shape[1]-1), 0).astype(np.int)
        mask[xx, yy] = 255
        yy = np.maximum(np.minimum(y-w, shape[1]-1), 0).astype(np.int)
        mask[xx, yy] = 255
    return mask


def debug_frame(main_img, wimg, wmasks, lmask, rmask, lanes_ransac):
    """Create a debug frame...
    """
    shape = main_img.shape[0:2]
    half_shape = (shape[0] // 2, shape[1] // 2)
    new_shape = (int(shape[0] * 1.5), int(shape[1] * 1.5), 3)
    dimg = np.zeros(new_shape, dtype=main_img.dtype)

    # Main image
    dimg[:shape[0], :shape[1]] = main_img

    # Masks images...
    l = 2
    offset = 0
    titles = ['RGB gradients mask',
              'HSV gradients mask']
    for i in range(l):
        img = np.copy(wimg)
        img = draw_mask(img, lmask, alpha=.9, beta=1., gamma=0., color=[20, 0, 0])
        img = draw_mask(img, rmask, alpha=1, beta=1., gamma=0., color=[0, 20, 0])
        img = draw_mask(img, wmasks[i], alpha=0.9, beta=1., gamma=0., color=[255, 255, 0])
        cv2.putText(img, titles[i], (50, 70), cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 2)

        dimg[shape[0]:, offset:offset+shape[1]//2] = cv2.resize(img, half_shape[::-1])
        offset += half_shape[1]

    # Lanes fits.
    l = min(len(lanes_ransac.w_fits), 3)
    offset = 0
    titles = ['Left -> Right RANSAC + L2 regr.',
              'Right -> Left RANSAC + L2 regr.',
              'Previous frame + L2 regr.']
    for i in range(l):
        w1, w2 = lanes_ransac.w_fits[i]
        X_lane, y1_lane, y2_lane = predict_lanes_w(w1, w2, wimg, reversed_x=True, normalised=True)
        x_lane = X_lane[:, 1]

        # Lanes predictions.
        left_mask = lane_mask(shape, x_lane, y1_lane, 8)
        right_mask = lane_mask(shape, x_lane, y2_lane, 8)

        dist_lanes = w2[0] - w1[0]
        curv1 = lane_curvature(w1)
        curv2 = lane_curvature(w2)

        img = np.copy(wimg)
        img = draw_mask(img, left_mask, alpha=.7, beta=1., gamma=0., color=[255, 0, 0])
        img = draw_mask(img, right_mask, alpha=1, beta=1., gamma=0., color=[0, 255, 0])

        # Add text...
        m1, m2 = lanes_ransac.inliers_masks[i]
        n1 = np.sum(m1)
        n2 = np.sum(m2)
        cv2.putText(img, titles[i], (50, 70), cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 2)
        cv2.putText(img, 'Inliers: %i | %i' % (n1, n2), (50, 120), cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 2)
        cv2.putText(img, 'Curvatures:  %.2f |  %.2f' % (curv1, curv2), (50, 170), cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 2)
        cv2.putText(img, 'W1:  %s' % w1, (50, 220), cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 2)
        cv2.putText(img, 'W2:  %s' % w2, (50, 270), cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 2)

        dimg[offset:offset+shape[0]//2:, shape[1]:] = cv2.resize(img, half_shape[::-1])
        offset += half_shape[0]

    return dimg
