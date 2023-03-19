import numpy as np
from ._numeric import numeric_fit
from ._frequentist import mle_fit
from ._caviar_function import adaptive, symmetric_abs_val, asymmetric_slope, igarch


class CaviarModel:
    def __init__(self, quantile, model, method='numeric', G=10, tol=1e-10):
        """
        :param: quantile (float): quantile value between 0 and 1 exclusively
        :param: model (str): type of CAViaR model
        :param: G (int): some positive number for the smoothen version of indicator function
        """
        self.beta = None
        self.p = None

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
            caviar = adaptive
        elif self.model == 'symmetric':
            caviar = symmetric_abs_val
        elif self.model == 'asymmetric':
            caviar = asymmetric_slope
        else:  # IGARCH
            caviar = igarch

        if self.method == 'numeric':
            return numeric_fit(returns, self.model, self.quantile, caviar, self.obj, self.tol)

        elif self.method == 'mle':
            return mle_fit(returns, self.model, self.quantile, caviar)
