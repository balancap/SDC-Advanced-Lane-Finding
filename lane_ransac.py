# coding: utf-8

# Author: Johannes Schönberger
#
# License: BSD 3 clause

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
# Ransac pre-fitting.
# =========================================================================== #
@numba.jit(nopython=True, nogil=True)
def inverse_4x4(m):
    """Inverse 4x4 matrix. Manual implementation!

    Allow comparison with numpy.inv, see if it is actually faster or the
    latter makes use of additional optimisations (MKL?)
    """
    mflat = m.reshape((m.size, ))
    minv = np.zeros_like(mflat)

    # Compute individual coefficient.
    minv[0] = \
        mflat[5] * mflat[10] * mflat[15] - \
        mflat[5] * mflat[11] * mflat[14] - \
        mflat[9] * mflat[6] * mflat[15] + \
        mflat[9] * mflat[7] * mflat[14] + \
        mflat[13] * mflat[6] * mflat[11] - \
        mflat[13] * mflat[7] * mflat[10]

    minv[4] = \
        -mflat[4] * mflat[10] * mflat[15] + \
        mflat[4] * mflat[11] * mflat[14] + \
        mflat[8] * mflat[6] * mflat[15] - \
        mflat[8] * mflat[7] * mflat[14] - \
        mflat[12] * mflat[6] * mflat[11] + \
        mflat[12] * mflat[7] * mflat[10]

    minv[8] = \
        mflat[4] * mflat[9] * mflat[15] - \
        mflat[4] * mflat[11] * mflat[13] - \
        mflat[8] * mflat[5] * mflat[15] + \
        mflat[8] * mflat[7] * mflat[13] + \
        mflat[12] * mflat[5] * mflat[11] - \
        mflat[12] * mflat[7] * mflat[9]

    minv[12] = \
        -mflat[4] * mflat[9] * mflat[14] + \
        mflat[4] * mflat[10] * mflat[13] + \
        mflat[8] * mflat[5] * mflat[14] - \
        mflat[8] * mflat[6] * mflat[13] - \
        mflat[12] * mflat[5] * mflat[10] + \
        mflat[12] * mflat[6] * mflat[9]

    minv[1] = \
        -mflat[1] * mflat[10] * mflat[15] + \
        mflat[1] * mflat[11] * mflat[14] + \
        mflat[9] * mflat[2] * mflat[15] - \
        mflat[9] * mflat[3] * mflat[14] - \
        mflat[13] * mflat[2] * mflat[11] + \
        mflat[13] * mflat[3] * mflat[10]

    minv[5] = \
        mflat[0] * mflat[10] * mflat[15] - \
        mflat[0] * mflat[11] * mflat[14] - \
        mflat[8] * mflat[2] * mflat[15] + \
        mflat[8] * mflat[3] * mflat[14] + \
        mflat[12] * mflat[2] * mflat[11] - \
        mflat[12] * mflat[3] * mflat[10]

    minv[9] = \
        -mflat[0] * mflat[9] * mflat[15] + \
        mflat[0] * mflat[11] * mflat[13] + \
        mflat[8] * mflat[1] * mflat[15] - \
        mflat[8] * mflat[3] * mflat[13] - \
        mflat[12] * mflat[1] * mflat[11] + \
        mflat[12] * mflat[3] * mflat[9]

    minv[13] = \
        mflat[0] * mflat[9] * mflat[14] - \
        mflat[0] * mflat[10] * mflat[13] - \
        mflat[8] * mflat[1] * mflat[14] + \
        mflat[8] * mflat[2] * mflat[13] + \
        mflat[12] * mflat[1] * mflat[10] - \
        mflat[12] * mflat[2] * mflat[9]

    minv[2] = \
        mflat[1] * mflat[6] * mflat[15] - \
        mflat[1] * mflat[7] * mflat[14] - \
        mflat[5] * mflat[2] * mflat[15] + \
        mflat[5] * mflat[3] * mflat[14] + \
        mflat[13] * mflat[2] * mflat[7] - \
        mflat[13] * mflat[3] * mflat[6]

    minv[6] = \
        -mflat[0] * mflat[6] * mflat[15] + \
        mflat[0] * mflat[7] * mflat[14] + \
        mflat[4] * mflat[2] * mflat[15] - \
        mflat[4] * mflat[3] * mflat[14] - \
        mflat[12] * mflat[2] * mflat[7] + \
        mflat[12] * mflat[3] * mflat[6]

    minv[10] = \
        mflat[0] * mflat[5] * mflat[15] - \
        mflat[0] * mflat[7] * mflat[13] - \
        mflat[4] * mflat[1] * mflat[15] + \
        mflat[4] * mflat[3] * mflat[13] + \
        mflat[12] * mflat[1] * mflat[7] - \
        mflat[12] * mflat[3] * mflat[5]

    minv[14] = \
        -mflat[0] * mflat[5] * mflat[14] + \
        mflat[0] * mflat[6] * mflat[13] + \
        mflat[4] * mflat[1] * mflat[14] - \
        mflat[4] * mflat[2] * mflat[13] - \
        mflat[12] * mflat[1] * mflat[6] + \
        mflat[12] * mflat[2] * mflat[5]

    minv[3] = \
        -mflat[1] * mflat[6] * mflat[11] + \
        mflat[1] * mflat[7] * mflat[10] + \
        mflat[5] * mflat[2] * mflat[11] - \
        mflat[5] * mflat[3] * mflat[10] - \
        mflat[9] * mflat[2] * mflat[7] + \
        mflat[9] * mflat[3] * mflat[6]

    minv[7] = \
        mflat[0] * mflat[6] * mflat[11] - \
        mflat[0] * mflat[7] * mflat[10] - \
        mflat[4] * mflat[2] * mflat[11] + \
        mflat[4] * mflat[3] * mflat[10] + \
        mflat[8] * mflat[2] * mflat[7] - \
        mflat[8] * mflat[3] * mflat[6]

    minv[11] = \
        -mflat[0] * mflat[5] * mflat[11] + \
        mflat[0] * mflat[7] * mflat[9] + \
        mflat[4] * mflat[1] * mflat[11] - \
        mflat[4] * mflat[3] * mflat[9] - \
        mflat[8] * mflat[1] * mflat[7] + \
        mflat[8] * mflat[3] * mflat[5]

    minv[15] = \
        mflat[0] * mflat[5] * mflat[10] - \
        mflat[0] * mflat[6] * mflat[9] - \
        mflat[4] * mflat[1] * mflat[10] + \
        mflat[4] * mflat[2] * mflat[9] + \
        mflat[8] * mflat[1] * mflat[6] - \
        mflat[8] * mflat[2] * mflat[5]

    det = mflat[0] * minv[0] + mflat[1] * minv[4] + mflat[2] * minv[8] + mflat[3] * minv[12]
    det = 1.0 / det

    for i in range(16):
        minv[i] = minv[i] * det
    minv = minv.reshape(m.shape)

    # if det == 0:
    #     return false;
    return minv


# =========================================================================== #
# Main Ransac implementation.
# =========================================================================== #
def _dynamic_max_trials(n_inliers, n_samples, min_samples, probability):
    """Determine number trials such that at least one outlier-free subset is
    sampled for the given inlier/outlier ratio.

    Parameters
    ----------
    n_inliers : int
        Number of inliers in the data.

    n_samples : int
        Total number of samples in the data.

    min_samples : int
        Minimum number of samples chosen randomly from original data.

    probability : float
        Probability (confidence) that one outlier-free sample is generated.

    Returns
    -------
    trials : int
        Number of trials.

    """
    inlier_ratio = n_inliers / float(n_samples)
    nom = max(_EPSILON, 1 - probability)
    denom = max(_EPSILON, 1 - inlier_ratio ** min_samples)
    if nom == 1:
        return 0
    if denom == 1:
        return float('inf')
    return abs(float(np.ceil(np.log(nom) / np.log(denom))))


class LanesRANSACRegressor(BaseEstimator, MetaEstimatorMixin, RegressorMixin):
    """RANSAC (RANdom SAmple Consensus) algorithm.

    RANSAC is an iterative algorithm for the robust estimation of parameters
    from a subset of inliers from the complete data set. More information can
    be found in the general documentation of linear models.

    A detailed description of the algorithm can be found in the documentation
    of the ``linear_model`` sub-package.

    Read more in the :ref:`User Guide <ransac_regression>`.

    Parameters
    ----------
    base_estimator : object, optional
        Base estimator object which implements the following methods:

        * `fit(X, y)`: Fit model to given training data and target values.
        * `score(X, y)`: Returns the mean accuracy on the given test data,
           which is used for the stop criterion defined by `stop_score`.
           Additionally, the score is used to decide which of two equally
           large consensus sets is chosen as the better one.

        If `base_estimator` is None, then
        ``base_estimator=sklearn.linear_model.LinearRegression()`` is used for
        target values of dtype float.

        Note that the current implementation only supports regression
        estimators.

    min_samples : int (>= 1) or float ([0, 1]), optional
        Minimum number of samples chosen randomly from original data. Treated
        as an absolute number of samples for `min_samples >= 1`, treated as a
        relative number `ceil(min_samples * X.shape[0]`) for
        `min_samples < 1`. This is typically chosen as the minimal number of
        samples necessary to estimate the given `base_estimator`. By default a
        ``sklearn.linear_model.LinearRegression()`` estimator is assumed and
        `min_samples` is chosen as ``X.shape[1] + 1``.

    residual_threshold : float, optional
        Maximum residual for a data sample to be classified as an inlier.
        By default the threshold is chosen as the MAD (median absolute
        deviation) of the target values `y`.

    is_data_valid : callable, optional
        This function is called with the randomly selected data before the
        model is fitted to it: `is_data_valid(X, y)`. If its return value is
        False the current randomly chosen sub-sample is skipped.

    is_model_valid : callable, optional
        This function is called with the estimated model and the randomly
        selected data: `is_model_valid(model, X, y)`. If its return value is
        False the current randomly chosen sub-sample is skipped.
        Rejecting samples with this function is computationally costlier than
        with `is_data_valid`. `is_model_valid` should therefore only be used if
        the estimated model is needed for making the rejection decision.

    max_trials : int, optional
        Maximum number of iterations for random sample selection.

    stop_n_inliers : int, optional
        Stop iteration if at least this number of inliers are found.

    stop_score : float, optional
        Stop iteration if score is greater equal than this threshold.

    stop_probability : float in range [0, 1], optional
        RANSAC iteration stops if at least one outlier-free set of the training
        data is sampled in RANSAC. This requires to generate at least N
        samples (iterations)::

            N >= log(1 - probability) / log(1 - e**m)

        where the probability (confidence) is typically set to high value such
        as 0.99 (the default) and e is the current fraction of inliers w.r.t.
        the total number of samples.

    residual_metric : callable, optional
        Metric to reduce the dimensionality of the residuals to 1 for
        multi-dimensional target values ``y.shape[1] > 1``. By default the sum
        of absolute differences is used::

            lambda dy: np.sum(np.abs(dy), axis=1)

        NOTE: residual_metric is deprecated from 0.18 and will be removed in 0.20
        Use ``loss`` instead.

    loss : string, callable, optional, default "absolute_loss"
        String inputs, "absolute_loss" and "squared_loss" are supported which
        find the absolute loss and squared loss per sample
        respectively.

        If ``loss`` is a callable, then it should be a function that takes
        two arrays as inputs, the true and predicted value and returns a 1-D
        array with the ``i``th value of the array corresponding to the loss
        on `X[i]`.

        If the loss on a sample is greater than the ``residual_threshold``, then
        this sample is classified as an outlier.

    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.

    Attributes
    ----------
    estimator_ : object
        Best fitted model (copy of the `base_estimator` object).

    n_trials_ : int
        Number of random selection trials until one of the stop criteria is
        met. It is always ``<= max_trials``.

    inlier_mask_ : bool array of shape [n_samples]
        Boolean mask of inliers classified as ``True``.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/RANSAC
    .. [2] http://www.cs.columbia.edu/~belhumeur/courses/compPhoto/ransac.pdf
    .. [3] http://www.bmva.org/bmvc/2009/Papers/Paper355/Paper355.pdf
    """

    def __init__(self, base_estimator=None, min_samples=None,
                 residual_threshold=None, is_data_valid=None,
                 is_model_valid=None, max_trials=100,
                 stop_n_inliers=np.inf, stop_score=np.inf,
                 stop_probability=0.99, residual_metric=None,
                 loss='absolute_loss', random_state=None):

        self.base_estimator = base_estimator
        self.min_samples = min_samples
        self.residual_threshold = residual_threshold
        self.is_data_valid = is_data_valid
        self.is_model_valid = is_model_valid
        self.max_trials = max_trials
        self.stop_n_inliers = stop_n_inliers
        self.stop_score = stop_score
        self.stop_probability = stop_probability
        self.residual_metric = residual_metric
        self.random_state = random_state
        self.loss = loss

    def fit(self, X1, y1, X2, y2, sample_weight=None):
        """Fit estimator using RANSAC algorithm.
        """
        # X = check_array(X, accept_sparse='csr')
        # y = check_array(y, ensure_2d=False)
        check_consistent_length(X1, y1)
        check_consistent_length(X2, y2)

        if self.base_estimator is not None:
            base_estimator1 = clone(self.base_estimator)
            base_estimator2 = clone(self.base_estimator)
        else:
            base_estimator1 = LinearRegression()
            base_estimator2 = LinearRegression()

        # assume linear model by default
        if self.min_samples is None:
            min_samples = X1.shape[1] + 1
        elif 0 < self.min_samples < 1:
            min_samples = np.ceil(self.min_samples * X1.shape[0])
        elif self.min_samples >= 1:
            if self.min_samples % 1 != 0:
                raise ValueError("Absolute number of samples must be an "
                                 "integer value.")
            min_samples = self.min_samples
        else:
            raise ValueError("Value for `min_samples` must be scalar and "
                             "positive.")
        if min_samples > X1.shape[0] or min_samples > X2.shape[0]:
            raise ValueError("`min_samples` may not be larger than number "
                             "of samples ``X1-2.shape[0]``.")

        if self.stop_probability < 0 or self.stop_probability > 1:
            raise ValueError("`stop_probability` must be in range [0, 1].")

        if self.residual_threshold is None:
            # MAD (median absolute deviation)
            residual_threshold = np.median(np.abs(y - np.median(y)))
        else:
            residual_threshold = self.residual_threshold

        if self.loss == "absolute_loss":
            if y1.ndim == 1:
                loss_function = lambda y_true, y_pred: np.abs(y_true - y_pred)
            else:
                loss_function = lambda \
                    y_true, y_pred: np.sum(np.abs(y_true - y_pred), axis=1)
        elif self.loss == "squared_loss":
            if y1.ndim == 1:
                loss_function = lambda y_true, y_pred: (y_true - y_pred) ** 2
            else:
                loss_function = lambda \
                    y_true, y_pred: np.sum((y_true - y_pred) ** 2, axis=1)
        elif callable(self.loss):
            loss_function = self.loss
        else:
            raise ValueError(
                "loss should be 'absolute_loss', 'squared_loss' or a callable."
                "Got %s. " % self.loss)


        random_state = check_random_state(self.random_state)
        try:  # Not all estimator accept a random_state
            base_estimator1.set_params(random_state=random_state)
            base_estimator2.set_params(random_state=random_state)
        except ValueError:
            pass

        estimator_fit_has_sample_weight = has_fit_parameter(base_estimator1,
                                                            "sample_weight")
        estimator_name = type(base_estimator1).__name__
        if (sample_weight is not None and not estimator_fit_has_sample_weight):
            raise ValueError("%s does not support sample_weight. Samples"
                             " weights are only used for the calibration"
                             " itself." % estimator_name)
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)

        # Best match variables.
        n_inliers_best1 = 0
        n_inliers_best2 = 0
        score_best1 = np.inf
        score_best2 = np.inf
        inlier_mask_best1 = None
        inlier_mask_best2 = None
        X1_inlier_best = None
        X2_inlier_best = None
        y1_inlier_best = None
        y2_inlier_best = None

        # number of data samples
        n_samples1 = X1.shape[0]
        sample_idxs1 = np.arange(n_samples1)
        n_samples2 = X2.shape[0]
        sample_idxs2 = np.arange(n_samples2)

        for self.n_trials_ in range(1, self.max_trials + 1):

            # Choose random sample sets 1 and 2
            subset_idxs1 = sample_without_replacement(n_samples1, min_samples,
                                                      random_state=random_state)
            X1_subset = X1[subset_idxs1]
            y1_subset = y1[subset_idxs1]
            subset_idxs2 = sample_without_replacement(n_samples2, min_samples,
                                                      random_state=random_state)
            X2_subset = X2[subset_idxs2]
            y2_subset = y2[subset_idxs2]

            # check if random sample set is valid
            if (self.is_data_valid is not None and not
                    self.is_data_valid(X1_subset, y1_subset, X2_subset, y2_subset)):
                continue

            # fit model for current random sample set
            if sample_weight is None:
                base_estimator1.fit(X1_subset, y1_subset)
                base_estimator2.fit(X2_subset, y2_subset)
            # else:
            #     base_estimator.fit(X_subset, y_subset,
            #                        sample_weight=sample_weight[subset_idxs])

            # check if estimated model is valid
            if (self.is_model_valid is not None and not
                    self.is_model_valid(base_estimator1, X1_subset, y1_subset,
                                        base_estimator2, X2_subset, y2_subset)):
                continue

            # Predictions on full dataset.
            y_pred1 = base_estimator1.predict(X1)
            y_pred2 = base_estimator2.predict(X2)

            residuals_subset1 = loss_function(y1, y_pred1)
            residuals_subset2 = loss_function(y2, y_pred2)

            # classify data into inliers and outliers
            inlier_mask_subset1 = residuals_subset1 < residual_threshold
            n_inliers_subset1 = np.sum(inlier_mask_subset1)
            inlier_mask_subset2 = residuals_subset2 < residual_threshold
            n_inliers_subset2 = np.sum(inlier_mask_subset2)

            # less inliers -> skip current random sample
            if n_inliers_subset1 + n_inliers_subset2 < n_inliers_best1 + n_inliers_best2:
                continue
            if n_inliers_subset1 == 0 or n_inliers_subset2 == 0:
                continue
                # raise ValueError("No inliers found, possible cause is "
                #     "setting residual_threshold ({0}) too low.".format(
                #     self.residual_threshold))

            # extract inlier data set
            inlier_idxs_subset1 = sample_idxs1[inlier_mask_subset1]
            X1_inlier_subset = X1[inlier_idxs_subset1]
            y1_inlier_subset = y1[inlier_idxs_subset1]

            inlier_idxs_subset2 = sample_idxs2[inlier_mask_subset2]
            X2_inlier_subset = X2[inlier_idxs_subset2]
            y2_inlier_subset = y2[inlier_idxs_subset2]

            # Score of inlier datasets
            score_subset1 = base_estimator1.score(X1_inlier_subset,
                                                  y1_inlier_subset)
            score_subset2 = base_estimator2.score(X2_inlier_subset,
                                                  y2_inlier_subset)

            # same number of inliers but worse score -> skip.
            if (n_inliers_subset1 + n_inliers_subset2 == n_inliers_best1 + n_inliers_best2
                    and score_subset1 + score_subset2 < score_best1 + score_best2):
                continue

            # save current random sample as best sample
            n_inliers_best1 = n_inliers_subset1
            score_best1 = score_subset1
            inlier_mask_best1 = inlier_mask_subset1
            X1_inlier_best = X1_inlier_subset
            y1_inlier_best = y1_inlier_subset

            n_inliers_best2 = n_inliers_subset2
            score_best2 = score_subset2
            inlier_mask_best2 = inlier_mask_subset2
            X2_inlier_best = X2_inlier_subset
            y2_inlier_best = y2_inlier_subset

            # break if sufficient number of inliers or score is reached
            if ((n_inliers_best1 >= self.stop_n_inliers
                    and n_inliers_best2 >= self.stop_n_inliers)
                    or (score_best1 >= self.stop_score and score_best2 >= self.stop_score)):
                    # or self.n_trials_
                    #    >= _dynamic_max_trials(n_inliers_best, n_samples,
                    #                           min_samples,
                    #                           self.stop_probability)):
                break

        # if none of the iterations met the required criteria
        if inlier_mask_best1 is None or inlier_mask_best2 is None :
            raise ValueError(
                "RANSAC could not find valid consensus set, because"
                " either the `residual_threshold` rejected all the samples or"
                " `is_data_valid` and `is_model_valid` returned False for all"
                " `max_trials` randomly ""chosen sub-samples. Consider "
                "relaxing the ""constraints.")

        # Estimate final model using all inliers
        base_estimator1.fit(X1_inlier_best, y1_inlier_best)
        base_estimator2.fit(X2_inlier_best, y2_inlier_best)

        self.estimator1_ = base_estimator1
        self.inlier_mask1_ = inlier_mask_best1
        self.estimator2_ = base_estimator2
        self.inlier_mask2_ = inlier_mask_best2
        return self

    def predict(self, X1, X2):
        """Predict using the estimated model.

        This is a wrapper for `estimator_.predict(X)`.

        Parameters
        ----------
        X : numpy array of shape [n_samples, n_features]

        Returns
        -------
        y : array, shape = [n_samples] or [n_samples, n_targets]
            Returns predicted values.
        """
        return self.estimator1_.predict(X1), self.estimator2_.predict(X2)

    def score(self, X1, y1, X2, y2):
        """Returns the score of the prediction.

        This is a wrapper for `estimator_.score(X, y)`.

        Parameters
        ----------
        X : numpy array or sparse matrix of shape [n_samples, n_features]
            Training data.

        y : array, shape = [n_samples] or [n_samples, n_targets]
            Target values.

        Returns
        -------
        z : float
            Score of the prediction.
        """
        return self.estimator1_.score(X1, y1) + self.estimator1_.score(X2, y2)
