#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
Created on 2024/1/17 16:00
@Author  : Tengdi Zheng
@File    : data_generation.py
"""


from util import *
RANDOM_STATE = 42


def ar_cov_mat(p, rho):
    """
    Generate the covariance matrix.

    Parameter
    ----------------
    p : number of variables

    rho : a constant
        Correlation coefficient.

    Returns
    ---------------
    Sigma : ndarray of (n_locations, n_locations)
        Covariance matrix.
    """
    Sigma = np.full((p, p), np.nan)
    for i in range(p):
        for j in range(p):
            Sigma[i, j] = rho ** abs(i - j)
            Sigma[j, i] = Sigma[i, j]
    return Sigma

def generate_one_point_data(N, beta, datatype, rho, sigma, p1, p2, p_noise):
    """
    Generate data for a single location.

    Parameter
    ----------------
    N : sample size for a single location

    beta : ndarray of shape (n_features, )
        The truth value of the coefficient for a single location.

    datatype : a string
        Different data type ('discrete', 'half', 'continue').

    rho : a constant
        Correlation coefficient.

    sigma : a constant
        Noise level.

    p1, p2, p_noise : the number of continuous variables, discrete variables, noise variables

    Returns
    ---------------
    dataset: ndarray of (n_m_samples, n_features + 1 (y))
        Dataset for a single location.
    """
    if datatype == 'discrete':
        Sigma = ar_cov_mat(p1 + p2 + p_noise, rho)
        data = np.random.multivariate_normal(np.zeros(p1 + p2 + p_noise), Sigma, N)
        data[:, p1:p1 + p2//2] = np.digitize(data[:, p1:p1 + p2//2], [np.quantile(data[:, p1:p1 + p2//2], 0.5)])
        data[:, p1 + p2//2:p1 + p2] = np.digitize(data[:, p1 + p2//2:p1 + p2], [np.quantile(data[:, p1 + p2//2:p1 + p2], 0.25)])
        y = np.c_[data[:, :p1 + p2], np.ones(len(data[:, :p1 + p2]))] @ beta + np.random.multivariate_normal(
            np.zeros(N), np.identity(N), N)[0,] * sigma
        dataset = np.concatenate((data.astype(np.float32), y.astype(np.float32).reshape(-1, 1)), axis=1)
        dataset = pd.DataFrame(dataset, columns=[*[f'x{i + 1}' for i in range(data.shape[1])], 'y'])

    elif datatype == 'half':
        Sigma = ar_cov_mat(p1 + p2 + p_noise, rho)
        data = np.random.multivariate_normal(np.zeros(p1 + p2 + p_noise), Sigma, N)
        data[:, p1:p1 + p2] = np.digitize(data[:, p1:p1 + p2], [np.quantile(data[:, p1:p1 + p2], 0.5)])
        y = np.c_[data[:, :p1 + p2], np.ones(len(data[:, :p1 + p2]))] @ beta + np.random.multivariate_normal(
            np.zeros(N), np.identity(N), N)[0,] * sigma
        dataset = np.concatenate((data.astype(np.float32), y.astype(np.float32).reshape(-1, 1)), axis=1)
        dataset = pd.DataFrame(dataset, columns=[*[f'x{i + 1}' for i in range(data.shape[1])], 'y'])

    elif datatype == 'continue':
        Sigma = ar_cov_mat(p1 + p2 + p_noise, rho)
        data = np.random.multivariate_normal(np.zeros(p1 + p2 + p_noise), Sigma, N)
        y = np.c_[data[:, :p1 + p2], np.ones(len(data[:, :p1 + p2]))] @ beta + np.random.multivariate_normal(
            np.zeros(N), np.identity(N), N)[0, ] * sigma
        dataset = np.concatenate((data.astype(np.float32), y.astype(np.float32).reshape(-1, 1)), axis=1)
        dataset = pd.DataFrame(dataset, columns=[*[f'x{i + 1}' for i in range(data.shape[1])], 'y'])
    return dataset

def generate_repeat_data(repeat_number, datatype, rho, sigma, signal_level, p1, p2, p_noise, l, g_coords, n):
    """
    Generate data for all locations.

    Parameter
    ----------------
    repeat_number : the number of times the experiment was repeated

    datatype : a string
        Different data type ('discrete', 'half', 'continue').

    rho : correlation coefficient

    sigma : noise level

    signal_level : a string
        Different signal level ('low', 'high').

    p1, p2, p_noise : the number of continuous variables, discrete variables, noise variables

    l : number of locations

    g_coords : ndarray of shape (n_location, 2 (lon, lat))
        Geographical coordinates.

    n : ndarray of (n_locations, )
        The number of samples from different locations.

    Returns
    ---------------
    repeat_data: ndarray of (n_samples, n_features + 3 (lon, lat, y))
        Datasets for all locations.

    beta : ndarray of (n_locations, n_features + 1)
        The truth values of the coefficients at different locations

    threshold : compressibility factor
    """
    beta = np.full((l, p1 + p2 + 1), np.nan)  # Initialize the regression coefficients
    a = np.linspace(1.5, 4, p1 + p2 + 1) * 10  # Get different features.
    np.random.seed(RANDOM_STATE)
    for feature in range(p1 + p2 + 1):  # The regression coefficients of all sites were obtained.
        beta[:, feature] = (np.sqrt(0.5 - (np.sqrt((g_coords[:, 0] - 0.5) ** 2 + (g_coords[:, 1] - 0.5) ** 2)) ** 2)) / a[feature]
        if signal_level == 'low':
            beta[np.random.choice(l, 2 * l // 10, replace=False), feature] = beta[np.random.choice(l, 2 * l // 10, replace=False), feature] * 3
        elif signal_level == 'high':
            beta[np.random.choice(l, 8 * l // 10, replace=False), feature] = beta[np.random.choice(l, 8 * l // 10, replace=False), feature] * 3
    threshold = np.min(beta[:, -2]) / 10
    repeat_data = []
    repeat_number = repeat_number
    for aa in range(repeat_number):
        np.random.seed(1314 + aa)
        simdat = []
        for i in range(l):
            datset = generate_one_point_data(n[i], beta[i, :], datatype, rho, sigma, p1, p2, p_noise)
            simdat.append(datset)
        data_with_loc = pd.DataFrame(
            columns=['location', *[i for i in simdat[0].columns if i not in ('location', 'y')], 'y'])
        for i in range(l):
            simdat[i]['location'] = i + 1
            simdat[i] = simdat[i][['location', *[i for i in simdat[i].columns if i not in ('location', 'y')], 'y']]
            data_with_loc = pd.concat([data_with_loc, simdat[i]], axis=0, ignore_index=True)
        data_generate = pd.DataFrame(columns=['loc_x', 'loc_y', *[i for i in simdat[i].columns if i not in ('location')]])
        for loc in range(l):
            data = data_with_loc[data_with_loc['location'] == loc + 1]
            data = data.drop('location', axis=1)
            data.insert(0, 'loc_x', g_coords[loc][0])
            data.insert(1, 'loc_y', g_coords[loc][1])
            data_generate = pd.concat([data_generate, data], axis=0, ignore_index=True)
        repeat_data.append(data_generate)
    return repeat_data, beta, threshold