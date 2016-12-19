"""
Classic Ridge regression adapted to lane detection: supposed two
lanes roughly parallel, and optimise the solution accordingly.
"""

from abc import ABCMeta, abstractmethod
import warnings
import numpy as np
import six

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils import check_random_state, check_array, check_consistent_length
from sklearn.linear_model.base import LinearModel

from scipy.optimize import minimize, rosen, rosen_der

FLOAT_DTYPES = np.float64


class LaneRidge(BaseEstimator):

    def __init__(self, alpha=None, w_reg=None, copy_X=True, max_iter=None, tol=1e-3,
                 random_state=None):
        self.alpha = alpha
        self.w_reg = w_reg
        self.copy_X = copy_X
        self.max_iter = max_iter or np.inf
        self.tol = tol
        self.random_state = random_state

    def fit(self, X1, y1, X2, y2, w0=None, sample_weight=None, disp=True):
        # Check a few things...
        assert X1.shape[0] == y1.shape[0], X2.shape[0] == y2.shape[0]
        assert X1.shape[1] == X2.shape[1]
        # Copy X values.
        if self.copy_X:
            X1 = np.copy(X1)
            X2 = np.copy(X2)

        n_samples1 = X1.shape[0]
        n_samples2 = X2.shape[0]
        n_features = X1.shape[1]

        if self.w_reg is None:
            self.w_reg = np.zeros((2*n_features, ), dtype=X1.dtype)
        if self.alpha is None:
            self.alpha = np.zeros((2*n_features, ), dtype=X1.dtype)

        # Pre-compute some quantities.
        X1X1 = X1.T @ X1
        X2X2 = X2.T @ X2
        X1y1 = X1.T @ y1
        X2y2 = X2.T @ y2

        # Loss function to minimize.
        def loss(w):
            w1 = w[:3] - w[3:]
            w2 = w[:3] + w[3:]

            l1 = np.sum(np.square(y1 - X1 @ w1)) * 1. / n_samples1
            l2 = np.sum(np.square(y2 - X2 @ w2)) * 1. / n_samples2
            # lreg = np.dot(self.alpha, np.square(w - self.w_reg))

            idx = self.alpha[3]
            kappa1 = (2*w1[2]) / ((1 + w1[1])**1.5 + 2*w1[2]*w[3])
            kappa2 = (2*w2[2]) / ((1 + w2[1])**1.5 - 2*w2[2]*w[3])
            lreg1 = self.alpha[0] * np.abs(kappa1 - kappa2)**idx

            lreg2 = self.alpha[1] * np.abs(w[4])**idx

            dist = np.abs(np.sum(w2) - np.sum(w1) - 2 * w[3])
            lreg3 = self.alpha[2] * dist**idx

            return l1 + l2 + lreg1 + lreg2 + lreg3

        def loss_gradient(w):
            wavg = w[:3]
            weps = w[3:]

            g = np.zeros_like(w)
            g[:3] = -2*(X1y1 + X2y2) - 2*(X1X1-X2X2) @ weps + 2*(X1X1+X2X2) @ wavg
            g[3:] = 2*(X1y1 - X2y2) - 2*(X1X1-X2X2) @ wavg + 2*(X1X1+X2X2) @ weps
            g += 2*self.alpha*(w - self.w_reg)
            return g

        # Minimize using BFGS (better method?)
        if w0 is None:
            w0 = np.zeros((2*n_features, ), dtype=X1.dtype)
        # options = {
        #     'disp': True,
        #     'gtol': self.tol,
        #     'maxiter': self.max_iter
        # }
        # res = minimize(loss, w0, method='BFGS', jac=loss_gradient, options=options)
        options = {
            'disp': True,
            'maxiter': self.max_iter
        }
        res = minimize(loss, w0, method='BFGS', options=options)

        # Save parameters.
        self.intercept_ = None
        self.coef_ = res.x

        # print(res.x)
        # print(res.success)

    def predict(self, X1, X2):
        """Predict y1 and y2 based on X1 and X2.
        """
        w = self.coef_
        w1 = w[:3] - w[3:]
        w2 = w[:3] + w[3:]
        return X1 @ w1, X2 @ w2

    def score(self, X1, y1, X2, y2):
        pass

