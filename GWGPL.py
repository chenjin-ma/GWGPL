#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
Created on 2024/1/17 14:23
@Author  : Tengdi Zheng
@File    : GWGPL.py
"""

from util import *
import itertools
from scipy.linalg import block_diag
import warnings

class GWGPL():
    def __init__(
            self,
            lamd=1e-3,
            tol=1e-3,
            max_iter=1000,
            verbose=False,
            threshold=0.001
    ):
        '''
        :param lamd: regularization parameter lambda
        :param tol: early stopping tol
        :param max_iter: maximum iterations s_max
        :param verbose: output result or not
        :param threshold: compressibility factor
        '''
        self.lamd = lamd
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.threshold = threshold

    def Diagonal_matrix_construction(self, X, y, Aerfa, g_coords):
        """
        Data conversion.

        Parameters
        ----------
        X : ndarray of (n_samples, n_features)
            Data.

        y : ndarray of shape (n_samples, )
            Target.

        Aerfa : ndarray of shape (n_locations, n_locations)
            Weight matrix.

        g_coords : ndarray of shape (n_locations, 2 (lon, lat))
            Geographical coordinates.

        Returns
        -------------
        X_total : ndarray of (n_samples * n_locations, n_locations * n_features)
            Data.

        X_b_total : ndarray of (n_samples * n_locations, n_locations)
            Data_0.

        y_total : ndarray of shape (n_samples * n_locations, )
            Target.
        """

        X_total, X_b_total, y_total = np.array([0]), np.array([0]), np.array([0]).reshape(-1, 1)
        for loc in range(len(g_coords)):
            aerfa = Aerfa.iloc[:, loc]
            w = []  # the geographical weight corresponding to y under this central point is obtained
            for i in range(len(X)):
                for j in range(len(aerfa)):
                    if X[i][0] == aerfa.index[j][0] and X[i][1] == aerfa.index[j][1]:
                        w.append(aerfa[aerfa.index[j]])
            aer = np.matrix(w).reshape(-1, 1)
            # aer = aer * (X.shape[0] / np.sum(aer))
            aer = np.power(aer, 0.5)
            XX, XX_b, yy = np.matrix(X[:, 2:]), np.matrix(np.ones(X[:, 2:].shape[0])).reshape(-1, 1), np.matrix(y).reshape(-1, 1)
            XX, XX_b, yy = np.multiply(aer, XX), np.multiply(aer, XX_b), np.multiply(aer, yy)
            X_total, X_b_total, y_total = block_diag(X_total, XX), block_diag(X_b_total, XX_b), np.concatenate((y_total, yy), axis=0)
        X_total, X_b_total, y_total = X_total[1:, 1:], X_b_total[1:, 1:], y_total[1:, :]

        return X_total, X_b_total, y_total

    def Group_identification(self, g_coords, p):
        """
        Group_identification.

        Parameters
        ----------------
        g_coords : ndarray of shape (n_locations, 2 (lon, lat))
            Geographical coordinates.

        p : number of variables

        Returns
        -------
        group_id:  ndarray of shape (p * l, )
            Group identification.
        """
        group_id = []
        for i in range(len(g_coords)):
            for j in range(p):
                group_id.append(j)  # same group has the same label
        return group_id

    def model(self, X, X_b, y, group_id, p, l):
        """
        Fit model with coordinate descent.

        Parameters
        ----------
        X : ndarray of (n_samples * n_locations, n_locations * n_features)
            Data.

        X_b : ndarray of (n_samples * n_locations, n_locations)
            Data_0.

        y : ndarray of shape (n_samples * n_locations, )
            Target.

        group_id : ndarray of shape (n_locations * n_features, )
            Group identification.

        p : number of variables

        l : number of locations

        Returns
        -------
        self : object
            Fitted estimator.

        """
        assert X.shape[1] == p * l, \
            'The row number of `X` must be the same as `p * l` '
        w, b = np.matrix(np.zeros((l * p, 1))), np.matrix(np.zeros(l)).reshape(-1, 1)
        r = y
        r_old, norm_old = r, np.Inf
        niter = itertools.count(1)

        Assit = [[] for row in range(l)]
        P_col = [[] for row in range(l)]
        nn, pp = X.shape[0], X.shape[1]
        for loc in range(l):
            X_l = X[loc * (nn//l): (loc+1) * (nn//l), loc * (pp//l): (loc+1) * (pp//l)]
            for cov in range(pp//l):
                Assit[loc].append((1/(X_l[:, cov].T @ X_l[:, cov]) * X_l[:, cov].T).reshape(1,-1))
                P_col[loc].append(X_l[:, cov].reshape(-1,1) @ Assit[loc][cov])

        for it in niter:
            norm = 0
            for cov in range(p):
                cov_group = [x for x, y in list(enumerate(group_id)) if y == cov]  # extract the index of the corresponding feature
                r_cov = r + X[:, cov_group] * w[cov_group]
                Pri, XiTri = 0, np.matrix(np.zeros(l)).reshape(-1, 1)
                for i in range(l):
                    Pri += np.linalg.norm(P_col[i][cov] @ r_cov[i * (nn//l): (i+1) * (nn//l)], ord=2)**2
                    XiTri[i, :] = Assit[i][cov] * r_cov[i * (nn//l): (i+1) * (nn//l)]
                Pri = np.sqrt(Pri)
                if Pri < self.lamd / 2:
                    w[cov_group] = np.matrix(np.zeros(l)).reshape(-1, 1)
                else:
                    w[cov_group] = (1 - self.lamd / (2 * Pri)) * XiTri
                r = r_cov - X[:, cov_group] * w[cov_group]
                norm += np.linalg.norm(X[:, cov_group] * w[cov_group], ord=2)

            r_b = r + X_b * b
            b = np.linalg.pinv(X_b.T @ X_b) * (X_b.T @ (y - X @ w))
            r = r_b - X_b * b
            delta_old = np.linalg.norm(r_old, ord=2) ** 2 + self.lamd * norm_old
            delta = np.linalg.norm(r, ord=2) ** 2 + self.lamd * norm

            if abs(delta_old - delta) < self.tol or it > self.max_iter:
                break
            else:
                r_old, norm_old = r, norm
                if self.verbose:
                    if it // 100 == 0:
                        print('Iteration: {}, delta = {}'.format(it, delta))

        coef = np.array(w).reshape(l, p)

        if self.threshold:
            coef_s = np.average(np.abs(coef),axis=0)
            coef_log = coef_s < self.threshold
            # coef[:, pd.DataFrame(coef_log).eq(True).all().values.tolist()] = 0
            coef[:, coef_log] = 0

        self.coef_ = coef.reshape(-1, 1)
        self.intercept_ = b
        return self

    def fit(self, X, y, Aerfa, g_coords, p, l):
        """
        Training model.

        Parameters
        ----------
        X : ndarray of (n_samples, n_features)
            Data.

        y : ndarray of shape (n_samples, )
            Target.

        Aerfa : ndarray of shape (n_locations, n_locations)
            Weight matrix.

        g_coords : ndarray of shape (n_locations, 2 (lon, lat))
            Geographical coordinates.

        p : number of variables

        l : number of locations


        Returns
        -------
        self : object
            Fitted estimator.

        Notes
        -----
        Coordinate descent is an algorithm that only one axis direction is
        optimized in each iteration and the values of other axes are fixed,
        so that the multi-variable optimization problem becomes a univariate
        optimization problem.

        In order to improve the calculation speed, some matrices that need to
        be reused are calculated in advance.
        """

        if self.lamd == 0 and sum(sum(np.triu(Aerfa, 1))):
            warnings.warn(
                "With lamda=0 and h<np.min(distance), You are advised to use "
                "the LinearRegression for each location. ",
                stacklevel=2
            )

        elif self.lamd == 0:
            warnings.warn(
                "With lamda=0, You are advised to use geographical weighting "
                "algorithm.",
                stacklevel=2,
            )

        elif sum(sum(np.triu(Aerfa, 1))) == 0:
            warnings.warn(
                "With h<np.min(distance), You are advised to use Lasso for "
                "each location.",
                stacklevel=2,
            )

        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # Make changes to the data
        X, X_b, y = \
            self.Diagonal_matrix_construction(X, y, Aerfa, g_coords)

        # The location identification of the corresponding variable group
        group_id = \
            self.Group_identification(g_coords, p)

        return self.model(X, X_b, y, group_id, p, l)

    def predict(self, X, loc, p):
        """
        Prediction.

        Parameter
        ----------------
        X : ndarray of (n_samples, n_features + 1)
            Data.

        loc : model location

        p : number of variables

        Returns
        ---------------
        y_hat : ndarray of (n_samples, )
            Predict result.
        """
        W = np.vstack((self.coef_[loc*p : (loc+1)*p, :], self.intercept_[loc]))
        return X @ W