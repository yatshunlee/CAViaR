# Author: Lee Yat Shun, Jasper
# Copyright (c) 2023 Lee Yat Shun, Jasper. All rights reserved.

import numpy as np
from ._numeric import numeric_fit
from ._frequentist import mle_fit
from ._caviar_function import adaptive, symmetric_abs_val, asymmetric_slope, igarch
from ._utils import plot_caviar, plot_news_impact_curve
from time import time


class CaviarModel:
    def __init__(self, quantile=0.05, model='symmetric', method='numeric', G=10, tol=1e-10):
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
        """
        self.beta = None
        self.p = None
        self.caviar = None

        self.G = G
        self.tol = tol

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

    def obj(self, beta, returns, quantile, caviar):
        """
        :param: beta (array-like): parameters of CAVIAR function
        :param: returns (array-like): a series of returns
        :param: quantile (float): a value between 0 and 1
        :param: caviar (callable function): a CAVIAR function
        :return: quantile regression loss
        """
        VaR = caviar(returns, beta, quantile)
        residuals = returns - VaR
        hit = self.quantile - (returns < VaR)
        T = len(returns)
        return residuals @ hit / T

    def fit(self, returns):
        """
        :param: returns (array-like): a series of returns
        """
        # computed the daily returns as 100 times the difference of the log of the prices.
        returns = np.array(returns) * 100

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
                                    self.tol)

        elif self.method == 'mle':
            self.beta = mle_fit(returns, 
                                self.model, 
                                self.quantile, 
                                self.caviar)
        
        # print statistics
        print('Final loss:', self.obj(self.beta,
                                      returns,
                                      self.quantile,
                                      self.caviar))
        print(f'Time taken(s): {time() - s:.2f}')
        
    def predict(self, returns):
        returns = np.array(returns) * 100
        VaR = self.caviar(returns, self.beta, self.quantile)
        return VaR
        
    
    def plot_caviar(self, returns):
        try:
            x_axis = returns.index
        except:
            x_axis = None
        returns = np.array(returns) * 100
        VaR = self.caviar(returns, self.beta, self.quantile)
        plot_caviar(returns, VaR, self.quantile, self.model, x_axis)
        
    def plot_news_impact_curve(self, VaR=-1.645):
        plot_news_impact_curve(self.beta, self.model, self.quantile, VaR, self.G)
        