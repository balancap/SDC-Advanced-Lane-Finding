# coding: utf-8

# Author: Johannes Schönberger
#
# License: BSD 3 clause

import copy
import numpy as np
import numba

from sklearn.base import BaseEstimator, MetaEstimatorMixin, RegressorMixin, clone
from sklearn.utils import check_random_state, check_array, check_consistent_length
from sklearn.utils.random import sample_without_replacement
from sklearn.utils.validation import check_is_fitted
from sklearn.linear_model.base import LinearRegression
from sklearn.utils.validation import has_fit_parameter

_EPSILON = np.spacing(1)


# =========================================================================== #
# Hand implementation of 3x3 and 4x4 inverse: ~3x faster than numpy.linalg.inv
# =========================================================================== #
@numba.jit(nopython=True, nogil=True)
def inverse_3x3(m):
    """Inverse 3x3 matrix. Manual implementation!

    Very basic benchmarks show it's ~3x faster than calling numpy inverse
    method. Nevertheless, I imagine that much better optimised version exist
    in the MKL or other library (using SIMD, AVX, and so on).
    I have no idea how far Numba+LLVM is able to go in terms of optimisation of
    this code.
    """
    mflat = m.reshape((m.size, ))
    minv = np.zeros_like(mflat)

    minv[0] = mflat[4] * mflat[8] - mflat[5] * mflat[7]
    minv[3] = -mflat[3] * mflat[8] + mflat[5] * mflat[6]
    minv[6] = mflat[3] * mflat[7] - mflat[4] * mflat[6]

    minv[1] = -mflat[1] * mflat[8] + mflat[2] * mflat[7]
    minv[4] = mflat[0] * mflat[8] - mflat[2] * mflat[6]
    minv[7] = -mflat[0] * mflat[7] + mflat[1] * mflat[6]

    minv[2] = mflat[1] * mflat[5] - mflat[2] * mflat[4]
    minv[5] = -mflat[0] * mflat[5] + mflat[2] * mflat[3]
    minv[8] = mflat[0] * mflat[4] - mflat[1] * mflat[3]

    det = mflat[0] * minv[0] + mflat[1] * minv[3] + mflat[2] * minv[6]
    # UGGGGGLLLLLLLLYYYYYYYYYY!
    if np.abs(det) <= _EPSILON:
        det = 1e-10

    det = 1.0 / det
    for i in range(9):
        minv[i] = minv[i] * det
    minv = minv.reshape((3, 3))
    return minv


@numba.jit(nopython=True, nogil=True)
def inverse_3x3_symmetric(m):
    """Inverse 3x3 symmetric matrix. Manual implementation!
    """
    mflat = m.reshape((m.size, ))
    minv = np.zeros_like(mflat)

    minv[0] = mflat[4] * mflat[8] - mflat[5] * mflat[7]
    minv[3] = -mflat[3] * mflat[8] + mflat[5] * mflat[6]
    minv[6] = mflat[3] * mflat[7] - mflat[4] * mflat[6]

    minv[1] = minv[3]
    minv[4] = mflat[0] * mflat[8] - mflat[2] * mflat[6]
    minv[7] = -mflat[0] * mflat[7] + mflat[1] * mflat[6]

    minv[2] = minv[6]
    minv[5] = minv[7]
    minv[8] = mflat[0] * mflat[4] - mflat[1] * mflat[3]

    det = mflat[0] * minv[0] + mflat[1] * minv[3] + mflat[2] * minv[6]
    # UGGGGGLLLLLLLLYYYYYYYYYY!
    if np.abs(det) <= _EPSILON:
        det = 1e-10

    det = 1.0 / det
    for i in range(9):
        minv[i] = minv[i] * det
    minv = minv.reshape((3, 3))
    return minv


# =========================================================================== #
# Ransac pre-fitting.
# =========================================================================== #
@numba.jit(nopython=True, nogil=True)
def numpy_sign(x):
    if x >= 0:
        return 1
    else:
        return -1

@numba.jit(nopython=True, nogil=True)
def is_model_valid(w, wrefs, bounds):
    """Check if a regression diffs model is valid, based on the coefficients.
    Use references coefficients to check w is inside valid bounds.
    Make two
    different types of checking: difference between left and right lanes AND
    individual bounds for every lanes.

    For bounds parameters: here is the second to last index meaning:
      0: Distance between origin points;
      1: Angle at the origin (in radian);
      2: Curvature (compute the relative difference between them);

    Params:
      w: Coefficient of the fit;
      wrefs: Array of reference coefficients;
      bounds: Array of bounds.
    Return
      Is it valid?
    """
    res = True
    n_refs = wrefs.shape[0]
    for i in range(n_refs):
        wref = wrefs[i]
        diffs = bounds[i]

        # Distance at the origin.
        dist = w[0] - wref[0]
        res = np.abs(dist) >= diffs[0, 0]
        res = res and np.abs(dist) <= diffs[0, 1]

        # Angle at the origin.
        theta = np.arcsin(w[1]) - np.arcsin(wref[1])
        res = res and np.abs(theta) >= diffs[1, 0]
        res = res and np.abs(theta) <= diffs[1, 1]

        # Relative curvature.
        a1b2 = np.abs(wref[2]) * (1 + w[1]**2)**1.5
        a2b1 = np.abs(w[2]) * (1 + wref[1]**2)**1.5
        s = a1b2 + a2b1
        if s > _EPSILON:
            rel_curv = (a1b2*numpy_sign(w[2]) - a2b1*numpy_sign(wref[2]) + 2*dist*np.abs(w[2]*wref[2])) / s
            res = res and np.abs(rel_curv) >= diffs[2, 0]
            res = res and np.abs(rel_curv) <= diffs[2, 1]
    return res


@numba.jit(nopython=True, nogil=True)
def lanes_ransac_prefit(X, y,
                        n_prefits, max_trials,
                        w_refs, is_valid_bounds):
    """Construct some pre-fits for Ransac regression.

    Namely: select randomly 4 points, fit a 2nd order curve and then check the
    validity of the fit. Stop when n_prefits have been found or max_trials done.
    Note: aim to be much more efficient and faster than standard RANSAC.
    Could be easily run in parallel on a GPU.

    Params:
      X and y: Points to fit;
      n_prefits: Number of pre-fits to generate;
      max_trials: Maximum number of trials. No infinity loop!
      w_refs: Coefficients used for checking validity.
      is_valid_bounds: Bounds used for checking validity.
    """
    min_prefits = 1
    is_valid_check = w_refs.size == 0

    shape = X.shape
    w_prefits = np.zeros((n_prefits, 3), dtype=X.dtype)

    i = 0
    j = 0
    it = 0
    idxes = np.arange(shape[0])

    # Add w references to prefits.
    i += 1
    # n_refs = w_refs.shape[0]
    # for k in range(n_refs):
    #     w_prefits[i] = w_refs[k]
    #     i += 1

    # Initial shuffling of points.
    # Note: shuffling should be more efficient than random picking.
    # Processor cache is used much more efficiently this way.
    np.random.shuffle(idxes)
    X = X[idxes]
    y = y[idxes]

    # Fill the pre-fit arrays...
    while i < n_prefits and (it < max_trials or i < min_prefits):
        # Sub-sampling 4 points.
        _X = X[j:j+4]
        _y = y[j:j+4]
        # Solve linear regression! Hard job :)
        _XT = _X.T
        w = inverse_3x3_symmetric(_XT @ _X) @ _XT @ _y
        # Is model basically valid? Then save it!
        if is_valid_check or is_model_valid(w, w_refs, is_valid_bounds):
            w_prefits[i] = w
            i += 1
        j += 1
        it += 1

        # Get to the end: reshuffle another time!
        if j == shape[0]-3:
            np.random.shuffle(idxes)
            X = X[idxes]
            y = y[idxes]
            j = 0
    # Resize if necessary.
    if i < n_prefits:
        w_prefits = w_prefits[:i]
    return w_prefits


def test_lanes_ransac_prefit(n_prefits=1000):
    """Basic test of the lanes RANSAC pre-fit.
    """
    n = n_prefits
    X = np.random.rand(n, 3)
    y = np.random.rand(n)
    w_refs = np.ones((1, 3), dtype=X.dtype)
    valid_bounds = np.ones((1, 3, 2), dtype=X.dtype)
    lanes_ransac_prefit(X, y, n_prefits, n_prefits, w_refs, valid_bounds)


# =========================================================================== #
# Linear Regression: some optimised methods.
# =========================================================================== #
@numba.jit(nopython=True, nogil=True)
def m_regression_exp(X, y, w0, scales):
    """M-estimator used to regularise predictions. Use exponential weights.
    Iterates over an array of scales.
    """
    w = np.copy(w0)
    XX = np.copy(X)
    yy = np.copy(y)
    weights = np.zeros_like(y)
    for s in scales:
        y_pred = X @ w
        weights = np.exp(-np.abs(y - y_pred)**2 / s**2)
        XX[:, 0] = X[:, 0] * weights
        XX[:, 1] = X[:, 1] * weights
        XX[:, 2] = X[:, 2] * weights
        XXT = XX.T
        yy = y * weights
        w = np.linalg.inv(XXT @ XX) @ XXT @ yy
    return w


@numba.jit(nopython=True, nogil=True)
def linear_regression_fit(X, y):
    """Linear Regression: fit X and y.
    Very basic implementation based on inversing X.T @ X. Enough in low
    dimensions.
    """
    XT = X.T
    w = np.linalg.inv(XT @ X) @ XT @ y
    return w


@numba.jit(nopython=True, nogil=True)
def linear_regression_predict(X, w):
    """Linear Regression: predicted y from X and w.
    """
    y_pred = X @ w
    return y_pred


@numba.jit(nopython=True, nogil=True)
def linear_regression_score(X, y, w, mask):
    """Linear Regression: score in interval [0,1]. Compute L2 norm and y
    variance to obtain the score.
    """
    y_pred = X @ w
    u = np.sum((y - y_pred)**2 * mask)
    v = np.sum((y - np.mean(y))**2 * mask)
    if v > _EPSILON:
        score = 1 - u / v
    else:
        score = -np.inf
    return score


# =========================================================================== #
# Ransac Regression: best fit selection.
# =========================================================================== #
@numba.jit(nopython=True, nogil=True)
def lanes_inliers(X, y, w, residual_threshold):
    """Compute the RANSAC inliers mask, based on a given threshold and
    regression coefficients.
    """
    y_pred = X @ w
    residuals_subset = np.abs(y - y_pred)
    inlier_mask_subset = residuals_subset < residual_threshold
    return inlier_mask_subset


@numba.jit(nopython=True, nogil=True)
def ransac_absolute_loss(y_true, y_pred):
    """Absolute loss!
    """
    return np.abs(y_true - y_pred)


@numba.jit(nopython=True, nogil=True)
def lanes_ransac_select_best(X, y, w_prefits, residual_threshold):
    """Select best pre-fit from a collection. Score them using the number
    of inliers: keep the one cumulating the highest number.
    """
    n_prefits = w_prefits.shape[0]

    # Best match variables.
    n_inliers_best = 0
    score_best = np.inf
    inlier_mask_best = (y == np.inf)
    best_w = w_prefits[0]

    # Number of data samples
    n_samples = X.shape[0]
    sample_idxs = np.arange(n_samples)
    for i in range(n_prefits):
        # Predictions on the dataset.
        w = w_prefits[i]
        y_pred = X @ w

        # Inliers / outliers masks
        residuals_subset = np.abs(y - y_pred)
        # classify data into inliers and outliers
        inlier_mask_subset = residuals_subset < residual_threshold
        n_inliers_subset = np.sum(inlier_mask_subset)

        # less inliers -> skip current random sample
        if n_inliers_subset < n_inliers_best:
            continue

        # Score of inlier datasets
        score_subset = linear_regression_score(X, y, w, inlier_mask_subset)
        # same number of inliers but worse score -> skip.
        if (n_inliers_subset == n_inliers_best and score_subset < score_best):
            continue

        # Save current random sample as best sample
        score_best = score_subset
        n_inliers_best = n_inliers_subset
        inlier_mask_best = inlier_mask_subset
        best_w = w

    return best_w, inlier_mask_best


# =========================================================================== #
# Main Ransac Regression class. Scikit-inspired implementation.
# =========================================================================== #
class SLanesRANSACRegressor(BaseEstimator, MetaEstimatorMixin, RegressorMixin):
    """RANSAC (RANdom SAmple Consensus) algorithm adapted to lanes detection.

    RANSAC is an iterative algorithm for the robust estimation of parameters
    from a subset of inliers from the complete data set. We adapt here this
    generic algorithm to out lanes finding problem.
    """

    def __init__(self,
                 residual_threshold=None,
                 n_prefits=1000,
                 max_trials=100,
                 w_refs_left=None,
                 w_refs_right=None,
                 is_valid_bounds_left=None,
                 is_valid_bounds_right=None,
                 l2_scales=None,
                 smoothing=1.,
                 stop_n_inliers=np.inf,
                 stop_score=np.inf,
                 stop_probability=0.99,
                 random_state=None):

        self.residual_threshold = residual_threshold

        self.n_prefits = n_prefits
        self.max_trials = max_trials

        if w_refs_left is None:
            self.w_refs_left = np.zeros((0, 3), dtype=np.float32)
            self.is_valid_bounds_left = np.zeros((0, 3, 2), dtype=np.float32)
        else:
            self.w_refs_left = w_refs_left
            self.is_valid_bounds_left = is_valid_bounds_left
        if w_refs_right is None:
            self.w_refs_right = np.zeros((0, 3), dtype=np.float32)
            self.is_valid_bounds_right = np.zeros((0, 3, 2), dtype=np.float32)
        else:
            self.w_refs_right = w_refs_right
            self.is_valid_bounds_right = is_valid_bounds_right

        self.l2_scales = l2_scales
        self.smoothing = smoothing

        self.stop_n_inliers = stop_n_inliers
        self.stop_score = stop_score
        self.stop_probability = stop_probability
        self.random_state = random_state

    def fit(self, X1, y1, X2, y2, left_right_bounds=None):
        """Fit estimator using RANSAC algorithm.

        Namely, the fit is done into two main steps:
        - pre-fitting: quickly select n_prefits configurations which seems
        suitable given topological constraints.
        - finding best fit: select the pre-fit with the maximum number of inliers
        as the best fit.

        Inputs:
          X1, y1: Left lane points (supposedly)
          X2, y2: Right lane points (supposedly)
        """
        check_consistent_length(X1, y1)
        check_consistent_length(X2, y2)

        # Assume linear model by default
        min_samples = X1.shape[1] + 1
        if min_samples > X1.shape[0] or min_samples > X2.shape[0]:
            raise ValueError("`min_samples` may not be larger than number "
                             "of samples ``X1-2.shape[0]``.")

        # Check additional parameters...
        if self.stop_probability < 0 or self.stop_probability > 1:
            raise ValueError("`stop_probability` must be in range [0, 1].")
        if self.residual_threshold is None:
            residual_threshold = np.median(np.abs(y - np.median(y)))
        else:
            residual_threshold = self.residual_threshold
        # random_state = check_random_state(self.random_state)

        # Collections...
        self.w_fits = []
        self.w_fits_l2 = []
        self.inliers_masks = []
        self.n_inliers = []

        # === Left lane, and then, right lane === #
        w_left_prefits = lanes_ransac_prefit(X1, y1,
                                             self.n_prefits,
                                             self.max_trials,
                                             self.w_refs_left,
                                             self.is_valid_bounds_left)
        (w_left1, in_mask_left1) = lanes_ransac_select_best(X1, y1,
                                                            w_left_prefits,
                                                            residual_threshold)
        n_inliers_left1 = np.sum(in_mask_left1)
        w_refs = np.vstack((self.w_refs_right, np.reshape(w_left1, (1, 3))))
        is_valid_bounds = np.vstack((self.is_valid_bounds_right, left_right_bounds))
        w_right_prefits = lanes_ransac_prefit(X2, y2,
                                              self.n_prefits,
                                              self.max_trials,
                                              w_refs,
                                              is_valid_bounds)
        (w_right1, in_mask_right1) = lanes_ransac_select_best(X2, y2,
                                                              w_right_prefits,
                                                              residual_threshold)
        n_inliers_right1 = np.sum(in_mask_right1)
        n_inliers1 = n_inliers_right1 + n_inliers_left1

        self.w_fits.append((w_left1, w_right1))
        self.n_inliers.append(n_inliers1)
        self.inliers_masks.append((in_mask_left1, in_mask_right1))

        # === Right lane and then left lane === #
        w_right_prefits = lanes_ransac_prefit(X2, y2,
                                              self.n_prefits,
                                              self.max_trials,
                                              self.w_refs_right,
                                              self.is_valid_bounds_right)
        (w_right2, in_mask_right2) = lanes_ransac_select_best(X2, y2,
                                                              w_right_prefits,
                                                              residual_threshold)
        n_inliers_right2 = np.sum(in_mask_right2)
        w_refs = np.vstack((self.w_refs_left, np.reshape(w_right2, (1, 3))))
        is_valid_bounds = np.vstack((self.is_valid_bounds_left, left_right_bounds))
        w_left_prefits = lanes_ransac_prefit(X1, y1,
                                             self.n_prefits,
                                             self.max_trials,
                                             w_refs,
                                             is_valid_bounds)
        (w_left2, in_mask_left2) = lanes_ransac_select_best(X1, y1,
                                                            w_left_prefits,
                                                            residual_threshold)
        n_inliers_left2 = np.sum(in_mask_left2)
        n_inliers2 = n_inliers_right2 + n_inliers_left2

        self.w_fits.append((w_left2, w_right2))
        self.n_inliers.append(n_inliers2)
        self.inliers_masks.append((in_mask_left2, in_mask_right2))

        # === Previous frame??? === #
        if self.w_refs_left.size > 0 and self.w_refs_right.size > 0:
            in_mask_left3 = lanes_inliers(X1, y1, self.w_refs_left[0], residual_threshold)
            in_mask_right3 = lanes_inliers(X2, y2, self.w_refs_right[0], residual_threshold)
            n_inliers3 = np.sum(in_mask_left3) + np.sum(in_mask_right3)

            self.w_fits.append((self.w_refs_left[0], self.w_refs_right[0]))
            self.n_inliers.append(n_inliers3)
            self.inliers_masks.append((in_mask_left3, in_mask_right3))

        # L2 regression regularisation of fits.
        self.w_fits_l2 = copy.deepcopy(self.w_fits)
        if self.l2_scales is not None:
            for i in range(len(self.w_fits)):
                w1, w2 = self.w_fits[i]
                w_left = m_regression_exp(X1, y1, w1, self.l2_scales)
                w_right = m_regression_exp(X2, y2, w2, self.l2_scales)

                in_mask_left = lanes_inliers(X1, y1, w_left, residual_threshold)
                in_mask_right = lanes_inliers(X2, y2, w_right, residual_threshold)
                n_inliers = np.sum(in_mask_left) + np.sum(in_mask_right)

                self.w_fits_l2[i] = (w_left, w_right)
                self.n_inliers[i] = n_inliers
                self.inliers_masks[i] = (in_mask_left, in_mask_right)

        # Best fit?
        idx = np.argmax(self.n_inliers)
        w_left, w_right = self.w_fits_l2[idx]
        in_mask_left, in_mask_right = self.inliers_masks[idx]

        # Smoothing.
        smoothing = self.smoothing
        if self.w_refs_left.size > 0 and self.w_refs_right.size > 0:
            w_left = smoothing * w_left + (1. - smoothing) * self.w_refs_left[0]
            w_right = smoothing * w_right + (1. - smoothing) * self.w_refs_right[0]

        self.w1_ = w_left
        self.w2_ = w_right

        # Set regression parameters.
        base_estimator1 = LinearRegression(fit_intercept=False)
        base_estimator1.coef_ = w_left
        base_estimator1.intercept_ = 0.0
        base_estimator2 = LinearRegression(fit_intercept=False)
        base_estimator2.coef_ = w_right
        base_estimator2.intercept_ = 0.0

        # Save final model parameters.
        self.estimator1_ = base_estimator1
        self.estimator2_ = base_estimator2

        self.inlier_mask1_ = in_mask_left
        self.inlier_mask2_ = in_mask_right

        # # Estimate final model using all inliers
        # # base_estimator1.fit(X1_inlier_best, y1_inlier_best)
        # # base_estimator2.fit(X2_inlier_best, y2_inlier_best)

        return self

    def predict(self, X1, X2):
        """Predict`lanes using the estimated model.

        Parameters
          X1, X2.
        Returns
          y1, y2
        """
        return X1 @ self.w1_, X2 @ self.w2_
        # return self.estimator1_.predict(X1), self.estimator2_.predict(X2)

    def score(self, X1, y1, X2, y2):
        """Returns the score of the prediction.
        """
        return self.estimator1_.score(X1, y1) + self.estimator1_.score(X2, y2)
