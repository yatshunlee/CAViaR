# Author: Lee Yat Shun, Jasper
# Copyright (c) 2023 Lee Yat Shun, Jasper. All rights reserved.

import numpy as np
from numpy.linalg import inv
from scipy.stats import chi2, norm

def compute_se_pval(beta, vc_matrix):
    """
    To compute the standard errors of betas as well as the p values
    """
    beta_standard_errors = np.diag(vc_matrix) ** 0.5
    beta_pvals = norm.cdf(-abs(beta) / beta_standard_errors)
    return beta_standard_errors, beta_pvals

def hit_func(returns, VaRs, quantile):
    """
    returns: Indicator(return_t - VaR_t) - quantile (array-like)
    """
    return (returns < VaRs) - quantile

def dq_test(in_sample_mode, model, returns, quantile, VaRs, D, gradient, in_T, LAGS=4):
    """
    Use Manganelli's matlab code as a reference
    
    :param: in_sample_mode (bool): True (in-sample) / False (out-of-sample)
    :param: model (str):
    :param: returns (np.array): returns
    :param: quantile (float)
    :param: VaRs (np.array):
    :param: D (np.array): matrix
    :param: gradient (np.array): gradient vector for out-of-sample mode
    :param: in_T (int). Default is the size of training samples based on Rubia, A., & Sanchis-Marco, L. (2013)
    :param: LAGS (int). Default is 4.
    """
    T = len(returns)
    returns = np.array(returns)
    hit = hit_func(returns, VaRs, quantile)
    
    # Compute the quantile residuals
    residuals = returns - VaRs

    # Set up the bandwidth for the KNN algorithm
    sorted_result = np.sort(abs(residuals))
    # k = 40 if quantile == 0.01 else 60
    # following this approach:
    # Rubia, A., & Sanchis-Marco, L. (2013). On downside risk predictability through liquidity and trading activity: A dynamic quantile approach
    # k = int(np.sqrt(in_T))
    k = int(np.sqrt(T))
    bandwidth = sorted_result[k]
    
    constant = np.ones(T - LAGS)
    HIT = hit[LAGS:]
    VaRs_forecast = VaRs[LAGS:]
    # y_lag = y[LAGS-1:-1]
    
    #      [ 0        ][ 1        ][ ... ][ LAGS - 1 ]
    # Z =  [ 1        ][ 2        ][ ... ][ LAGS     ]
    #      [ ...      ][ ...      ][ ... ][ ...      ]
    #      [ T-LAGS-1 ][ T-LAGS   ][ ... ][ T-2      ], where these are the indices
    
    Z = np.zeros((T - LAGS, LAGS))
    for i in range(LAGS):
        Z[:, i] = hit[i:T-LAGS+i]
    
    # estimate the matrices for in-sample DQ test
    if in_sample_mode:
        X_in = Z
        XHNABLA = np.zeros((X_in.shape[1], gradient.shape[1]))
        NABLA = gradient[LAGS:, :]
        for i in range(1, X_in.shape[0]):
            if abs(residuals[i]) > bandwidth:
                continue
            XHNABLA += X_in[[i], :].T @ gradient[[i], :]
    
        XHNABLA = XHNABLA / (2 * bandwidth * T)

        M = X_in.T - XHNABLA @ inv(D) @ NABLA.T

        # compute the DQ tests
        DQ_stat_in = HIT.T @ X_in @ inv(M @ M.T) @ X_in.T @ HIT / (quantile * (1 - quantile))
        DQ_pval_in = chi2.cdf(DQ_stat_in, df=X_in.shape[1])
        return DQ_pval_in 
        
    else:
        X_out = np.c_[constant, VaRs_forecast, Z]
        DQ_stat_out = HIT.T @ X_out @ inv(X_out.T @ X_out) @ X_out.T @ HIT / (quantile * (1 - quantile))
        DQ_pval_out = chi2.cdf(DQ_stat_out, df=X_out.shape[1])
        return DQ_pval_out

def variance_covariance(beta, model, T, returns, quantile, VaRs, G):
    """
    Use Manganelli's matlab code as a reference
    
    :param: beta (np.array): fitted beta
    :param: model (str): one of CAViaR models
    :param: T (int): number of obs.
    :param: returns (np.array): returns
    :param: quantile (float): fitted quantile
    :param: VaRs (np.array): estimated value at risk
    :param: G (positive integer): for the sigmoid function in the adaptive CAViaR model
    """
    # Compute the quantile residuals
    residuals = returns - VaRs
    
    # Set up the bandwidth for the KNN algorithm by Engle and  Manganelli (2004)
    sorted_result = np.sort(abs(residuals))
    # k = 40 if quantile == 0.01 else 60
    # following this approach:
    # Rubia, A., & Sanchis-Marco, L. (2013). On downside risk predictability through liquidity and trading activity: A dynamic quantile approach
    k = int(np.sqrt(T))
    bandwidth = sorted_result[k]
    t = 0
    
    # initialize vectors
    derivative1 = np.zeros((T, 1))
    derivative2 = np.zeros((T, 1))
    derivative3 = np.zeros((T, 1))
    derivative4 = np.zeros((T, 1))
    
    D = np.zeros((beta.shape[0], beta.shape[0]))
    A = np.copy(D)
    
    if model == 'adaptive':
        # f_t = f_t-1 + b1 * ([1 + exp(G * (y_t-1 - f_t-1))]^-1 - quantile)
        gradient = np.zeros((T, beta.shape[0]))
        for i in range(1, T):
            # let sigmoid(x) = 1 / (1 + e^x)
            # d(sigmoid(x)) / dbeta = (sigmoid(x) ** 2 - sigmoid(x)) * dx/dx
            sigmoid = 1 / (1 + np.exp(G * (returns[i-1] - VaRs[i-1])))
            derivative1[i] = derivative1[i-1] + (
                (sigmoid - quantile) +
                beta[0] * (sigmoid ** 2 - sigmoid) * G * (- derivative1[i-1])
            )
            
            gradient[i, 0] = derivative1[i]
            
            A = A + gradient[[i], :].T @ gradient[[i], :]
            
            if abs(residuals[i]) <= bandwidth:
                t += 1
                D = D + gradient[[i], :].T @ gradient[[i], :]
    
    elif model == 'asymmetric':
        # f_t = b1 + b2 * f_t-1 + b3 * max(y_t-1, 0) + b4 * min(y_t-1, 0)
        gradient = np.zeros((T, beta.shape[0]))
        for i in range(1, T):
            derivative1[i] = 1 + beta[1] * derivative1[i-1]
            derivative2[i] = VaRs[i-1] + beta[1] * derivative2[i-1]
            derivative3[i] = max(returns[i-1], 0) + beta[1] * derivative3[i-1]
            derivative4[i] = min(returns[i-1], 0) + beta[1] * derivative4[i-1]
            
            gradient[i, 0] = derivative1[i]
            gradient[i, 1] = derivative2[i]
            gradient[i, 2] = derivative3[i]
            gradient[i, 3] = derivative4[i]
            
            A = A + gradient[[i], :].T @ gradient[[i], :] # this one should be a matrix
            
            if abs(residuals[i]) <= bandwidth:
                t += 1
                D = D + gradient[[i], :].T @ gradient[[i], :] # this one should be a matrix
    
    elif model == 'symmetric':
        gradient = np.zeros((T, beta.shape[0]))
        
        for i in range(1, T):
            derivative1[i] = 1 + beta[1] * derivative1[i-1]
            derivative2[i] = VaRs[i-1] + beta[1] * derivative2[i-1]
            derivative3[i] = abs(returns[i-1]) + beta[1] * derivative3[i-1]
            
            gradient[i, 0] = derivative1[i]
            gradient[i, 1] = derivative2[i]
            gradient[i, 2] = derivative3[i]
            
            A = A + gradient[[i], :].T @ gradient[[i], :] # this one should be a matrix
            
            if abs(residuals[i]) <= bandwidth:
                t += 1
                D = D + gradient[[i], :].T @ gradient[[i], :] # this one should be a matrix
                
    elif model == 'igarch':
        # f_t = (b1 + b2 * f_t-1 ** 2 + b3 * y_t-1 ** 2) ** 0.5
        gradient = np.zeros((T, beta.shape[0]))
        
        for i in range(1, T):
            derivative1[i] = (
                (1 + beta[1] * 2 * VaRs[i-1] * derivative1[i-1]) / (2 * VaRs[i])
            )
            derivative2[i] = (
                (VaRs[i-1] ** 2 + beta[1] * 2 * VaRs[i-1] * derivative2[i-1]) / (2 * VaRs[i])
            )
            derivative3[i] = (
                (returns[i-1] ** 2 + beta[1] * 2 * VaRs[i-1] * derivative3[i-1]) / (2 * VaRs[i])
            )
            
            gradient[i, 0] = derivative1[i]
            gradient[i, 1] = derivative2[i]
            gradient[i, 2] = derivative3[i]
            
            A = A + gradient[[i], :].T @ gradient[[i], :] # this one should be a matrix
            
            if abs(residuals[i]) <= bandwidth:
                t += 1
                D = D + gradient[[i], :].T @ gradient[[i], :] # this one should be a matrix
                
    else:
        raise ValueError('Wrong model!')
        
    t_std_err = t # check the bandwidth
    A = A / T
    D = D / (2 * bandwidth * T)

    vc_matrix = (quantile * (1 - quantile) / T) * (inv(D) @ A @ inv(D))
    
    return vc_matrix, D, gradient