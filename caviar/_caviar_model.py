import numpy as np
from ._numeric import numeric_fit
from ._frequentist import mle_fit
from ._caviar_function import adaptive, symmetric_abs_val, asymmetric_slope, igarch


class CaviarModel:
    def __init__(self, quantile=0.05, model='asymmetric', method='numeric', G=10, tol=1e-10):
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

    def obj(self, betas, returns, quantile, caviar):
        """
        :param: betas (array-like): parameters of CAVIAR function
        :param: returns (array-like): a series of returns
        :param: caviar (callable function): a CAVIAR function
        :return: quantile regression loss
        """
        sigmas = caviar(returns, betas, quantile)
        dev = returns - sigmas
        e = np.where(dev < 0, (self.quantile - 1) * dev, self.quantile * dev)
        return np.sum(e)

    def fit(self, log_returns):
        """
        :param: log_returns (array-like): a series of log returns
        """
        returns = np.array(log_returns) * 100

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

        if self.method == 'numeric':
            self.beta = numeric_fit(returns, self.model, self.quantile, self.caviar, self.obj, self.tol)

        elif self.method == 'mle':
            self.beta = mle_fit(returns, self.model, self.quantile, self.caviar)
