# Author: Lee Yat Shun, Jasper
# Copyright (c) 2023 Lee Yat Shun, Jasper. All rights reserved.

import numpy as np


def get_empirical_quantile(returns, quantile, until_first=300):
    """
    :param: returns (array): a series of daily returns
    :param: quantile (float): a value between 0 and 1
    :param: until_first (int): To compute the VaR series with the CAViaR models, we initialize f1(β)
                               to the empirical θ-quantile of the first 300 observations.
                               Default is 300 (days).
    :returns: VaR
    """
    return np.quantile(returns[:until_first], quantile)
    

def adaptive(returns, beta, quantile, G=10):
    """
    :param: returns (array): a series of daily returns
    :param: beta (array): a series of coefficients
    :param: quantile (float): a value between 0 and 1
    :param: G (int): some positive number for the smoothen version of indicator function
    :returns: VaR
    """
    b1 = beta[0]
    VaR = np.zeros_like(returns)
    VaR[0] = get_empirical_quantile(returns, quantile)
    for t in range(1, len(VaR)):
        VaR[t] = VaR[t - 1] + b1 * (
                1 / (1 + np.exp(G * (returns[t - 1] - VaR[t - 1]))) - quantile
        )
    return VaR


def symmetric_abs_val(returns, beta, quantile):
    """
    :param: returns (array): a series of daily returns
    :param: beta (array): a series of coefficients
    :param: quantile (float): a value between 0 and 1
    :returns: VaR
    """
    b1, b2, b3 = beta
    VaR = np.zeros_like(returns)
    VaR[0] = get_empirical_quantile(returns, quantile)
    for t in range(1, len(VaR)):
        VaR[t] = b1 + b2 * VaR[t - 1] + b3 * abs(returns[t - 1])
    return VaR


def asymmetric_slope(returns, beta, quantile):
    """
    :param: returns (array): a series of daily returns
    :param: beta (array): a series of coefficients
    :param: quantile (float): a value between 0 and 1
    :returns: VaR
    """
    b1, b2, b3, b4 = beta
    VaR = np.zeros_like(returns)
    VaR[0] = get_empirical_quantile(returns, quantile)
    for t in range(1, len(VaR)):
        VaR[t] = b1 + b2 * VaR[t - 1] + b3 * max(returns[t - 1], 0) + b4 * min(returns[t - 1], 0)
    return VaR


def igarch(returns, beta, quantile):
    """
    Notice that the sigma here is negative root of the sqrt term
    
    :param: returns (array): a series of daily returns
    :param: beta (array): a series of coefficients
    :param: quantile (float): a value between 0 and 1
    :returns: VaR
    """
    b1, b2, b3 = beta
    VaR = np.zeros_like(returns)
    VaR[0] = get_empirical_quantile(returns, quantile)
    for t in range(1, len(VaR)):
        VaR[t] = (b1 + b2 * VaR[t - 1] ** 2 + b3 * returns[t - 1] ** 2) ** 0.5
        if quantile < 0.5:
            VaR[t] *= -1
    return VaR


# def get_VaR(returns, beta, quantile, model):
#     VaR = np.zeros_like(returns)
#     VaR[0] = get_empirical_quantile(returns, quantile)
    
#     if model == 'symmetric':
#         b1, 
#     elif model == 'asymmetric':
#     elif model == 'adaptive':
#     elif model == 'igarch':
#     else:
#         raise ValueError('Wrong model')