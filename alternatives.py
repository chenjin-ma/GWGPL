#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
Created on 2024/1/17 14:36
@Author  : Tengdi Zheng
@File    : alternatives.py
"""

from util import *
import itertools

class one_point_estimation():
    def __init__(self, add_weight=False,
                 lamd=1e-3, tol=1e-3, max_iter=300, threshold=0.001):
        """
        Compared_method
        :param add_weight: whether to perform weighted processing
        :param lamd: regularization parameter lambda
        :param tol: early stopping tol
        :param max_iter: maximum iterations s_max
        :param threshold: compressibility factor
        """
        self.lamd = lamd
        self.tol = tol
        self.max_iter = max_iter
        self.add_weight = add_weight
        self.threshold = threshold

    def Lasso(self, X, y):
        """
        Fit model with Lasso.

        Parameters
        ----------
        X : ndarray of (n_samples, n_features)
            Data.

        y : ndarray of shape (n_samples, )
            Target.

        Returns
        -------
        self : object
            Fitted estimator.

        """
        b = 0
        rss = lambda X, y, w, b: np.linalg.norm(y - X * w - b, ord=2) ** 2 + self.lamd * np.linalg.norm(w, ord=1)
        # Initialization regression coefficient w
        n, p = X.shape
        w = np.matrix(np.zeros((p, 1)))
        r = rss(X, y, w, b)
        # Coordinate descent method was used to optimize the regression coefficient w
        niter = itertools.count(1)
        for it in niter:

            for k in range(p):
                # Calculate the constant values z_k and p_k
                z_k = np.linalg.norm(X[:, k], ord=2) ** 2
                p_k = 0
                for i in range(n):
                    p_k += X[i, k] * (y[i, 0] - sum([X[i, j] * w[j, 0] for j in range(p) if j != k]) - b)
                p_k = p_k
                if p_k < -self.lamd / 2:
                    w_k = (p_k + self.lamd / 2) / z_k
                elif p_k > self.lamd / 2:
                    w_k = (p_k - self.lamd / 2) / z_k
                else:
                    w_k = 0
                w[k, 0] = w_k

            b = np.sum(y - X @ w) / n
            r_prime = rss(X, y, w, b)
            delta = abs(r_prime - r)
            r = r_prime
            # print('Iteration: {}, delta = {}'.format(it, delta))
            if delta < self.tol or it > self.max_iter:
                # print("Converged. itr={}".format(it))
                break


        self.coef_ = np.array(w).reshape(1, -1)
        for i in range(self.coef_.shape[1]):
            if abs(self.coef_[:, i]) < self.threshold:
                self.coef_[:, i] = 0
            else:
                continue
        self.intercept_ = b
        return self

    def GWL(self, X, X_b, y, aer):
        """
        Fit model with geographically weighted lasso (GWL).

        Parameters
        ----------
        X : ndarray of (n_samples, n_features)
            Data.

        X_b : ndarray of (n_samples, )
            Data_0.

        y : ndarray of shape (n_samples, )
            Target.

        aer : ndarray of shape (n_samples, )
            Wight matrix.

        Returns
        -------
        self : object
            Fitted estimator.

        """
        rss = lambda X, y, X_b, w, b: (y - X * w - X_b * b).T * (y - X * w - X_b * b) + self.lamd * np.linalg.norm(w, ord=1)

        n, p = X.shape
        w = np.matrix(np.zeros((p, 1)))
        b = 0
        r = rss(X, y, X_b, w, b)

        niter = itertools.count(1)
        for it in niter:
            for k in range(p):
                z_k = np.linalg.norm(X[:, k], ord=2) ** 2
                p_k = 0
                for i in range(n):
                    p_k += X[i, k] * (
                            y[i, 0] - sum([X[i, j] * w[j, 0] for j in range(p) if j != k]) - X_b[i, 0] * b)
                p_k = p_k
                if p_k < -self.lamd / 2:
                    w_k = (p_k + self.lamd / 2) / z_k
                elif p_k > self.lamd / 2:
                    w_k = (p_k - self.lamd / 2) / z_k
                else:
                    w_k = 0
                w[k, 0] = w_k
            b = np.sum(np.multiply(aer, y) - np.multiply(aer, X) @ w) / np.sum(np.power(aer, 2))
            r_prime = rss(X, y, X_b, w, b)
            delta = abs(r_prime - r)
            r = r_prime
            # if self.verbose and it % self.verbose_interval == 0:
            # print('Iteration: {}, delta = {}'.format(it, delta))
            if delta < self.tol or it > self.max_iter:
                # print("Converged. itr={}".format(it))
                break
        self.coef_ = np.array(w).reshape(1, -1)
        for i in range(self.coef_.shape[1]):
            if abs(self.coef_[:, i]) < self.threshold:
                self.coef_[:, i] = 0
            else:
                continue
        self.intercept_ = b
        return self


    def fit(self, X, y, g_coords, loc, aerfa):
        """
       Training model.

       Parameters
       ----------
       X : ndarray of (n_samples, n_features)
           Data.

       y : ndarray of shape (n_samples, )
           Target.

       g_coords : ndarray of shape (n_locations, 2 (lon, lat))
           Geographical coordinates.

       loc : model location

       aerfa : ndarray of shape (n_locations, )
           Weight matrix.


       Returns
       -------
       self : object
           Different fitted estimators.
        """
        if isinstance(X, pd.DataFrame): X = X.values
        if isinstance(y, pd.Series): y = y.values

        if not self.add_weight and self.lamd == 0:
            '''
            Training with OLS.
            '''
            X = add_intercept(X)
            beta = Least_squares(X, y, g_coords, loc)
            self.coef_ = beta[:-1, :].reshape(1, -1)
            self.intercept_ = beta[-1, :]
            return self


        if self.add_weight and self.lamd == 0:
            '''
            Training with GWR.
            '''
            X, w = add_intercept(X), []
            for i in range(len(X)):
                for j in range(len(aerfa)):
                    if X[i][0] == aerfa.index[j][0] and X[i][1] == aerfa.index[j][1]:
                        w.append(aerfa[aerfa.index[j]])
            beta = localWeightRegression(X, y, w)
            self.coef_ = beta[:-1, :].reshape(1, -1)
            self.intercept_ = beta[-1, :]
            return self


        if not self.add_weight and self.lamd:
            '''
            Training with Lasso.
            '''
            XX, yy = X[list((X[:, 0] == g_coords[loc][0]) & (X[:, 1] == g_coords[loc][1])), :][:, 2:], \
                     y[list((X[:, 0] == g_coords[loc][0]) & (X[:, 1] == g_coords[loc][1]))]
            X, y = np.matrix(XX), np.matrix(yy)
            y = y.reshape(-1, 1)
            self.Lasso(X, y)


        if self.add_weight and self.lamd != 0:
            '''
            Training with GWL.
            '''
            w = []
            for i in range(len(X)):
                for j in range(len(aerfa)):
                    if X[i][0] == aerfa.index[j][0] and X[i][1] == aerfa.index[j][1]:
                        w.append(aerfa[aerfa.index[j]])
            aer = np.matrix(w).reshape(-1, 1)
            aer = np.power(aer, 0.5)
            X, y, X_b = np.matrix(X[:, 2:]), np.matrix(y), np.ones(X.shape[0]).reshape(-1, 1)
            y = y.reshape(-1, 1)
            X, y, X_b = np.multiply(aer, X), np.multiply(aer, y), np.multiply(aer, X_b)
            self.GWL(X, X_b, y, aer)


    def predict(self, X):
        """
        Prediction.

        Parameter
        ----------------
        X : ndarray of (n_samples, n_features + 1)
            Data.

        Returns
        ---------------
        y_hat : ndarray of (n_sample, )
            Predict result.
        """
        W = np.hstack((self.coef_, self.intercept_.reshape(-1, 1)))
        return X @ W.reshape(-1, 1)