# # Author: Lee Yat Shun, Jasper
# # Copyright (c) 2023 Lee Yat Shun, Jasper. All rights reserved.

# import numpy as np
# from numpy.linalg import inv
# from scipy.stats import chi2, binom_test


# # def binomial_test(returns, VaRs, quantile):
# #     """
# #     null hypothesis that the probability of success/failure in a Bernoulli experiment is p.
    
# #     :params: returns (array):
# #     :params: VaRs (array):
# #     :params: quantile (float):
# #     :returns: two-sided binomial test
# #     """
# #     k = count_violations(returns, VaRs) # number of failures
# #     n = len(returns) # num of total observations
# #     return binom_test(k, n, p=quantile)

# # def vrate(returns, VaRs):
# #     """
# #     :params: returns (array):
# #     :params: VaRs (array):
# #     returns: VRate (%): the violation rate
# #     """
# #     return count_violations(returns, VaRs)/returns.shape[0]
    
# # def count_violations(returns, VaRs):
# #     """
# #     :params: returns (array):
# #     :params: VaRs (array):
# #     :returns: number of violations
# #     """
# #     returns = np.array(returns)*100
# #     return np.sum(returns < VaRs)
    
# def dq_test(in_sample_mode, model, T, returns, quantile, VaR, hit, D, gradient, LAGS=4):
#     """
#     Use Manganelli's matlab code as a reference
#      The null hypothesis states that the current VaR violations are uncorrelated with past violations.
    
#     :param: in_sample_mode (bool): True (in-sample) / False (out-of-sample)
#     :param: model (str):
#     :param: T (int): number of in-sample obs.
#     :param: returns (np.array): returns
#     :param: quantile (float)
#     :param: VaR (np.array):
#     :param: hit (np.array):
#     :param: D (np.array): matrix
#     :param: gradient (np.array): gradient vector for out-of-sample mode
#     """
#     returns = np.array(returns) * 100
    
#     # Compute the quantile residuals
#     residuals = returns - VaR

#     # Set up the bandwidth for the KNN algorithm
#     sorted_result = np.sort(residuals)
#     k = 40 if quantile == 0.01 else 60
#     bandwidth = - sorted_result[k]
    
#     constant = np.ones(T - LAGS)
#     HIT = hit[LAGS:]
#     VaR_forecast = VaR[LAGS:]
#     # y_lag = y[LAGS-1:-1]
    
#     """
#          [ 0        ][ 1        ][ ... ][ LAGS - 1 ]
#     Z =  [ 1        ][ 2        ][ ... ][ LAGS     ], where these are the indices
#          [ ...      ][ ...      ][ ... ][ ...      ]
#          [ T-LAGS-1 ][ T-LAGS   ][ ... ][ T-2      ]
#     """
#     Z = np.zeros((T - LAGS, LAGS))
#     for i in range(LAGS):
#         Z[:, i] = hit[i:T-LAGS+i]
    
#     # estimate the matrices for in-sample DQ test
#     if in_sample_mode:
#         X_in = Z
#         XHNABLA = np.zeros((X_in.shape[1], gradient.shape[1]))
#         NABLA = gradient[LAGS:, :]
#         for i in range(1, X_in.shape[0]):
#             if abs(residuals[i]) > bandwidth:
#                 continue
#             XHNABLA += X_in[[i], :].T @ gradient[[i], :]
    
#         XHNABLA = XHNABLA / (2 * bandwidth * T)

#         M = X_in.T - XHNABLA @ inv(D) @ NABLA.T

#         # compute the DQ tests
#         DQ_stat_in = HIT.T @ X_in @ inv(M @ M.T) @ X_in.T @ HIT / (quantile * (1 - quantile))
#         DQ_pval_in = chi2.cdf(DQ_stat_in, df=X_in.shape[1])
#         return DQ_pval_in 
        
#     else:
#         X_out = np.c_[constant, VaR_forecast, Z]
#         DQ_stat_out = HIT.T @ X_out @ inv(X_out.T @ X_out) @ X_out.T @ HIT / (quantile * (1 - quantile))
#         DQ_pval_out = chi2.cdf(DQ_stat_out, df=X_out.shape[1])
#         return DQ_pval_out

# # def variance_covariance(returns, fitted_model):
#     # T = len(returns)
#     # beta = fitted_model.beta
#     # quantile = fitted_model.quantile
#     # VaR = fitted_model.predict(returns)
#     # returns = np.array(returns) * 100
# def variance_covariance(beta, model, T, returns, quantile, VaR, G=10):
#     """
#     Use Manganelli's matlab code as a reference
    
#     :param: beta (np.array)
#     :param: model (str)
#     :param: T (int): number of obs.
#     :param: returns (np.array): returns
#     :param: quantile (float)
#     :param: VaR (np.array):
#     """
#     returns = np.array(returns) * 100
#     # Compute the quantile residuals
#     residuals = returns - VaR
    
#     # Set up the bandwidth for the KNN algorithm
#     sorted_result = np.sort(residuals)
#     k = 40 if quantile == 0.01 else 60
#     bandwidth = - sorted_result[k]
#     t = 0
    
#     # initialize vectors
#     derivative1 = np.zeros((T, 1))
#     derivative2 = np.zeros((T, 1))
#     derivative3 = np.zeros((T, 1))
#     derivative4 = np.zeros((T, 1))
    
#     D = np.zeros((beta.shape[0], beta.shape[0]))
#     A = np.copy(D)
    
#     if model == 'adaptive':
#         # f_t = f_t-1 + b1 * ([1 + exp(G * (y_t-1 - f_t-1))]^-1 - quantile)
#         gradient = np.zeros((T, beta.shape[0]))
#         for i in range(1, T):
#             # let sigmoid(x) = 1 / (1 + e^x)
#             # d(sigmoid(x)) / dbeta = (sigmoid(x) ** 2 - sigmoid(x)) * dx/dx
#             sigmoid = 1 / (1 + np.exp(G * (returns[i-1] - VaR[i-1])))
#             derivative1[i] = derivative1[i-1] + (
#                 (sigmoid - quantile) +
#                 beta[0] * (sigmoid ** 2 - sigmoid) * G * (- derivative1[i-1])
#             )
            
#             gradient[i, 0] = derivative1[i]
            
#             A = A + gradient[[i], :].T @ gradient[[i], :]
            
#             if abs(residuals[i]) <= bandwidth:
#                 t += 1
#                 D = D + gradient[[i], :].T @ gradient[[i], :]
    
#     elif model == 'asymmetric':
#         # f_t = b1 + b2 * f_t-1 + b3 * max(y_t-1, 0) + b4 * min(y_t-1, 0)
#         gradient = np.zeros((T, beta.shape[0]))
#         for i in range(1, T):
#             derivative1[i] = 1 + beta[1] * derivative1[i-1]
#             derivative2[i] = VaR[i-1] + beta[1] * derivative2[i-1]
#             derivative3[i] = max(returns[i-1], 0) + beta[1] * derivative3[i-1]
#             derivative4[i] = min(returns[i-1], 0) + beta[1] * derivative4[i-1]
            
#             gradient[i, 0] = derivative1[i]
#             gradient[i, 1] = derivative2[i]
#             gradient[i, 2] = derivative3[i]
#             gradient[i, 3] = derivative4[i]
            
#             A = A + gradient[[i], :].T @ gradient[[i], :] # this one should be a matrix
            
#             if abs(residuals[i]) <= bandwidth:
#                 t += 1
#                 D = D + gradient[[i], :].T @ gradient[[i], :] # this one should be a matrix
    
#     elif model == 'symmetric':
#         gradient = np.zeros((T, beta.shape[0]))
        
#         for i in range(1, T):
#             derivative1[i] = 1 + beta[1] * derivative1[i-1]
#             derivative2[i] = VaR[i-1] + beta[1] * derivative2[i-1]
#             derivative3[i] = abs(returns[i-1]) + beta[1] * derivative3[i-1]
            
#             gradient[i, 0] = derivative1[i]
#             gradient[i, 1] = derivative2[i]
#             gradient[i, 2] = derivative3[i]
            
#             A = A + gradient[[i], :].T @ gradient[[i], :] # this one should be a matrix
            
#             if abs(residuals[i]) <= bandwidth:
#                 t += 1
#                 D = D + gradient[[i], :].T @ gradient[[i], :] # this one should be a matrix
                
#     elif model == 'igarch':
#         # f_t = (b1 + b2 * f_t-1 ** 2 + b3 * y_t-1 ** 2) ** 0.5
#         gradient = np.zeros((T, beta.shape[0]))
        
#         for i in range(1, T):
#             derivative1[i] = (
#                 (1 + beta[1] * 2 * VaR[i-1] * derivative1[i-1]) / (2 * VaR[i])
#             )
#             derivative2[i] = (
#                 (VaR[i-1] ** 2 + beta[1] * 2 * VaR[i-1] * derivative2[i-1]) / (2 * VaR[i])
#             )
#             derivative3[i] = (
#                 (returns[i-1] ** 2 + beta[1] * 2 * VaR[i-1] * derivative3[i-1]) / (2 * VaR[i])
#             )
            
#             gradient[i, 0] = derivative1[i]
#             gradient[i, 1] = derivative2[i]
#             gradient[i, 2] = derivative3[i]
            
#             A = A + gradient[[i], :].T @ gradient[[i], :] # this one should be a matrix
            
#             if abs(residuals[i]) <= bandwidth:
#                 t += 1
#                 D = D + gradient[[i], :].T @ gradient[[i], :] # this one should be a matrix
                
#     else:
#         raise ValueError('Wrong model!')
        
#     t_std_err = t # check the bandwidth
#     A = A / T
#     D = D / (2 * bandwidth * T)
    
#     vc_matrix = (quantile * (1 - quantile) / T) * (inv(D) @ A @ inv(D))
    
#     return vc_matrix, D, gradient