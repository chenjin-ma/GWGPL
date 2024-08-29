#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2024/1/17 15:26
@Author  : Tengdi Zheng
@File    : util.py
"""


import numpy as np
import pandas as pd

def bw_selection(g_coords):
    """
    Get bw_min and bw_max.

    Parameters
    ----------
    g_coords : ndarray of shape (n_locations, 2 (lon, lat))
            Geographical coordinates.

    Returns
    -------
    return:  two constants bw_min and bw_max

    """

    bb = np.zeros((len(g_coords), len(g_coords)))
    for i in range(len(g_coords)):
        for j in range(len(g_coords)):
            bb[i][j] = pow(pow(g_coords[i][0] - g_coords[j][0], 2)+pow(g_coords[i][1] - g_coords[j][1], 2), 0.5)
    return np.min(bb), np.max(bb)


def add_intercept(X):
    """
    Add intercept to X.

    Parameters
    ----------
     X : ndarray of (n_samples, n_features)
            Data.

    Returns
    -------
    return:  ndarray of (n_samples, n_features + 1)
            [X, X_0]
    """
    return np.c_[X, np.ones(len(X))]



def Kernel(g_coords, gwr_bw, kernel):
    """
    Kernel_function.

    Parameters
    ----------
    g_coords : ndarray of shape (n_locations, 2 (lon, lat))
        Geographical coordinates.
        
    gwr_bw : a constant
        Bandwidth.
    
    kernel : a string
        Specify different kernel functions ('threshold', 'bi-square', 'gaussian', 'exponential').

    Returns
    -------
    return:  ndarray of shape (n_locations, n_locations)
            Weight matrix.
    """
    aerfa = np.zeros((len(g_coords), len(g_coords)))
    for i in range(len(g_coords)):
        for j in range(len(g_coords)):
            aerfa[i][j] = pow(pow(g_coords[i][0] - g_coords[j][0], 2)+pow(g_coords[i][1] - g_coords[j][1], 2), 0.5)
            if aerfa[i][j] > gwr_bw:
                aerfa[i][j] = 0
            else:
                if kernel == 'threshold': aerfa[i][j] = 1
                if kernel == 'bi-square': aerfa[i][j] = pow(1 - pow((aerfa[i][j] / gwr_bw), 2), 2)
                if kernel == 'gaussian': aerfa[i][j] = np.exp(-0.5 * pow((aerfa[i][j] / gwr_bw), 2))
                if kernel == 'exponential': aerfa[i][j] = np.exp(-aerfa[i][j] / gwr_bw)
    return pd.DataFrame(aerfa, index=[list(g_coords[:, 0]), list(g_coords[:, 1])],
                         columns=[list(g_coords[:, 0]), list(g_coords[:, 1])])

def localWeightRegression(X, y, wt):
    """
    Geographic weighted regression (GWR).

    Parameters
    ----------
    X : ndarray of (n_samples, n_features)
        Data.

    y : ndarray of shape (n_samples, )
        Target.

    wt : ndarray of shape (n_samples, )
        The weighted size of each sample.

    Returns
    -------------
    beta : ndarray of shape (n_features + 1, )
        The final estimated coefficient value.
    """
    w = np.diag(np.array(wt))
    beta = np.linalg.pinv(X[:, 2:].T @ w @ X[:, 2:]) @ (X[:, 2:].T @ w @ y)
    beta = beta.reshape(len(beta), 1)
    return beta

def Least_squares(X, y, g_coords, loc):
    """
    Ordinary least squares regression (OLS).

    Parameters
    ----------
    X : ndarray of (n_samples, n_features)
        Data.

    y : ndarray of shape (n_samples, )
        Target.

    g_coords : ndarray of shape (n_locations, 2 (lon, lat))
        Geographical coordinates.

    loc : model location

    Returns
    -------------
    beta : ndarray of shape (n_features + 1, )
        The final estimated coefficient value.
    """
    X_train_lsq = X[list((X[:, 0] == g_coords[loc][0]) & (X[:, 1] == g_coords[loc][1])), :][:, 2:]
    y_train_lsq = y[list((X[:, 0] == g_coords[loc][0]) & (X[:, 1] == g_coords[loc][1]))]
    return np.linalg.pinv((X_train_lsq.T.dot(X_train_lsq))).dot(X_train_lsq.T).dot(y_train_lsq).reshape(-1, 1)