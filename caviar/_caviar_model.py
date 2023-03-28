# Author: Lee Yat Shun, Jasper
# Copyright (c) 2023 Lee Yat Shun, Jasper. All rights reserved.

import numpy as np
import pandas as pd
from ._numeric import numeric_fit
from ._frequentist import mle_fit
from ._caviar_function import adaptive, symmetric_abs_val, asymmetric_slope, igarch
from ._dq_test import compute_se_pval, variance_covariance, dq_test
from ._utils import plot_caviar, plot_news_impact_curve
from ._exceptions import InputSizeError, NotFittedError
from scipy.stats import norm
from time import time


class CaviarModel:
    def __init__(self, quantile=0.05, model='symmetric', method='numeric', G=10, tol=1e-10, LAGS=4):
        """
        CaviarModel is a class for estimating Conditional Autoregressive Value at Risk (CAViaR) models.
        
        :param: quantile (float): Quantile value between 0 and 1 exclusively. Default is 0.05.
        :param: model (str): Type of CAViaR model. Model must be one of {"adaptive", "symmetric", "asymmetric", "igarch"}.
                             Default is "asymmetric", i.e., asymmetric slope.
        :param: method (str): Estimation method. Must be one of {"numeric (Engle & Manganelli, 2004)",
                             "mle (Maximum Likelihood Estimation)"}.
                             Default is "numeric".
        :param: G (int): Smoothen version of the indicator function. Some positive number. Default is 10.
        :param: tol (float): Tolerance level for optimization. Default is 1e-10.
        :param: LAGS (int): Default is 4.
        """
        if G != 10:
            raise ValueError('Currently only support G = 10')
        
        self.beta = None
        self.p = None
        self.caviar = None
        self.VaR0_in = None
        self.VaR0_out = None
        
        self.vc_matrix = None
        self.D = None
        self.gradient = None

        self.G = G
        self.tol = tol
        self.LAGS = LAGS

        if 0 < quantile < 1:
            self.quantile = quantile
        else:
            raise ValueError('Quantile must be within 0, 1 exclusively.')

        if model in ['adaptive', 'symmetric', 'asymmetric', 'igarch']:
            self.model = model
        else:
            raise ValueError('Model must be one of {"adaptive", "symmetric", "asymmetric", "igarch"}')

        if method in ['numeric', 'mle']:
            self.method = method
        else:
            raise ValueError(
                'Method must be one of {"numeric (Engle & Manganelli, 2004)", "mle (Maximum Likelihood Estimation)"}'
            )
            
    def __repr__(self):
        return (f"CaviarModel(quantile={self.quantile}, model={self.model}, "
                f"method={self.method}, G={self.G}, tol={self.tol}, LAGS={self.LAGS})")
    
    def get_empirical_quantile(self, returns, quantile, until_first=300):
        """
        :param: returns (array): a series of daily returns
        :param: quantile (float): a value between 0 and 1
        :param: until_first (int): To compute the VaR series with the CAViaR models, we initialize f1(β)
                                   to the empirical θ-quantile of the first 300 observations.
                                   Default is 300 (days).
        :returns: VaR
        """
        return np.quantile(returns[:until_first], quantile)

    def obj(self, beta, returns, quantile, caviar, VaR0):
        """
        :param: beta (array-like): parameters of CAVIAR function
        :param: returns (array-like): a series of returns from day 0 to T
        :param: quantile (float): a value between 0 and 1
        :param: caviar (callable function): a CAVIAR function
        :return: quantile regression loss
        """
        # VaR from day 0 to T+1
        VaRs = caviar(returns, beta, quantile, VaR0, self.G)
        residuals = returns - VaRs[:-1]
        hit = self.quantile - (returns < VaRs[:-1])
        T = len(returns)
        return residuals @ hit / T

    def fit(self, returns):
        """
        :param: returns (array-like): a series of returns (100x)
        """
        if len(returns) < 300:
            raise InputSizeError('The size of return array must not be less than 300.')
        
        returns = np.array(returns)
        # starting point VaR_0 = unconditional sampling quantile
        self.VaR0_in = self.get_empirical_quantile(returns, self.quantile)
        
        # select the CAViaR function
        # symmetric and igarch: 3 betas; asymmetric: 4 betas; adaptive: 1 beta
        if self.model == 'adaptive':
            self.caviar = adaptive
        elif self.model == 'symmetric':
            self.caviar = symmetric_abs_val
        elif self.model == 'asymmetric':
            self.caviar = asymmetric_slope
        else:  # IGARCH
            self.caviar = igarch
            
        s = time()
        if self.method == 'numeric':
            self.beta = numeric_fit(returns,
                                    self.model,
                                    self.quantile,
                                    self.caviar,
                                    self.obj,
                                    self.tol,
                                    self.VaR0_in)

        elif self.method == 'mle':
            self.beta = mle_fit(returns, 
                                self.model, 
                                self.quantile, 
                                self.caviar,
                                self.VaR0_in,
                                self.G)
        
        # print statistics
        print('Final loss:', self.obj(self.beta,
                                      returns,
                                      self.quantile,
                                      self.caviar,
                                      self.VaR0_in))
        
        # To compute the variance and covariance matrix
        T = len(returns)
        VaRs = self.predict(returns, self.VaR0_in)
        self.VaR0_out = VaRs[-1]
        
        self.vc_matrix, self.D, self.gradient = variance_covariance(
            self.beta, self.model, T, returns, self.quantile, VaRs[:-1], self.G
        )
        
        # To compute the standard errors of betas as well as the p values
        self.beta_standard_errors, self.beta_pvals = compute_se_pval(self.beta, self.vc_matrix)
        
        print(f'Time taken(s): {time() - s:.2f}')
        
    def summary(self):
        """
        showing the pvalue and standard error of beta
        """
        if self.beta is None:
            msg = ('This CaviarModel instance is not fitted yet. '
                   'Call "fit" with appropriate arguments before using this estimator.')
            raise NotFittedError(msg)
            
        beta_df = pd.DataFrame({
            'coefficient': self.beta,
            'S.E. of beta': self.beta_standard_errors,
            'pval of beta': self.beta_pvals
        })

        beta_df.index = [f'beta{i+1}' for i in range(len(self.beta))]
        return beta_df
        
    def predict(self, returns, VaR0=None):
        """
        :param: returns (array-like): a series of returns
        :param: VaR0 (float): Initial VaR0. Default is the VaR0 (out-of-samples)
        """
        if self.beta is None:
            msg = ('This CaviarModel instance is not fitted yet. '
                   'Call "fit" with appropriate arguments before using this estimator.')
            raise NotFittedError(msg)
            
        if VaR0 is None:
            VaR0 = self.VaR0_out
        returns = np.array(returns)
        VaRs = self.caviar(returns, self.beta, self.quantile, VaR0, self.G)
        return VaRs
    
#     def forecast(self, return_ytd, VaR_ytd):
#         """
#         predict today's VaR (unknown)
#         :param: return_ytd (float): return yesterday
#         :param: VaR_ytd (float): VaR yesterday
#         :returns: VaR forecast today
#         """
#         if self.beta is None:
#             msg = ('This CaviarModel instance is not fitted yet. '
#                    'Call "fit" with appropriate arguments before using this estimator.')
#             raise NotFittedError(msg)
            
#         if self.model == 'adaptive':
#             b1 = self.beta[0]
#             return VaR_ytd + b1 * (
#                 1 / (1 + np.exp(self.G * (return_ytd - VaR_ytd))) - self.quantile
#             )
        
#         elif self.model == 'symmetric':
#             b1, b2, b3 = self.beta
#             return b1 + b2 * VaR_ytd + b3 * abs(return_ytd)
        
#         elif self.model == 'asymmetric':
#             b1, b2, b3, b4 = self.beta
#             return b1 + b2 * VaR_ytd + b3 * max(return_ytd, 0) + b4 * min(return_ytd, 0)
        
#         else:  # IGARCH
#             b1, b2, b3 = self.beta
#             VaR = (b1 + b2 * VaR_ytd ** 2 + b3 * return_ytd ** 2) ** 0.5
#             return - VaR
        
    def dq_test(self, returns, test_mode):
        """
        :param: returns (array-like):
        :param: test_mode (str): either 'in' or 'out' => 'in samples' or 'out of samples'
        """
        VaRs = self.predict(returns)
        if test_mode == 'in':
            return dq_test(True, self.model, returns, self.quantile, VaRs[:-1], self.D, self.gradient, self.LAGS)
        elif test_mode == 'out':
            return dq_test(False, self.model, returns, self.quantile, VaRs[:-1], self.D, self.gradient, self.LAGS)
        else:
            raise ValueError('Test mode must be one of {"in", "out"}')
    
    def plot_caviar(self, returns, mode):
        """
        plot the positive VaR and the violations in the fitting process
        :param: returns (array-like):
        """
        if self.beta is None:
            msg = ('This CaviarModel instance is not fitted yet. '
                   'Call "fit" with appropriate arguments before using this estimator.')
            raise NotFittedError(msg)
            
        if mode == 'in':
            VaR0 = self.VaR0_in
        elif mode == 'out':
            VaR0 = self.VaR0_out
        else:
            raise ValueError('mode must be either "in" or "out".')
        
        try:
            x_axis = returns.index
        except:
            x_axis = None
        returns = np.array(returns)
        VaRs = self.caviar(returns, self.beta, self.quantile, VaR0, self.G)
        plot_caviar(returns, VaRs[:-1], self.quantile, self.model, x_axis)
        
    def plot_news_impact_curve(self, VaR=-1.645):
        """
        visualizing how return_t-1 affect the VaR_t
        by fixing the VaR_t-1
        where news = return
        """
        if self.beta is None:
            msg = ('This CaviarModel instance is not fitted yet. '
                   'Call "fit" with appropriate arguments before using this estimator.')
            raise NotFittedError(msg)
            
        plot_news_impact_curve(self.beta, self.model, self.quantile, VaR, self.G)
        