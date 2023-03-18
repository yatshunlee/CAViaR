import numpy as np
from scipy.optimize import minimize

class GarchModel:
    def __init__(self, p, q):
        self.p = p
        self.q = q
        
    def arch(self, params, returns):
        """return a series of sigma2"""
        omega = params[0]
        alphas = params[1:]
                
        T = len(returns)
        
        return2s = returns ** 2
        
        sigma2s = np.zeros(T)
        sigma2s[0:self.p] = np.var(returns)
        for t in range(self.p, T):
            sigma2s[t] = omega + alphas @ return2s[t-self.p:t]
        
        if np.any(sigma2s<=0):
            raise ValueError('sigma_2_t <= 0: insignificant p')
            
        return sigma2s
    
    def garch(self, params, returns):
        """return a series of sigma2"""
        omega = params[0]
        alphas = params[1:1+self.p]
        betas = params[1+self.p:]
                
        T = len(returns)
        
        return2s = returns ** 2
        
        sigma2s = np.zeros(T)
        sigma2s[0:self.p] = np.var(returns)
        for t in range(self.p, T):
            sigma2s[t] = omega + alphas @ return2s[t-self.p:t]
        
        if np.any(sigma2s<=0):
            raise ValueError('sigma_2_t <= 0: insignificant p')
            
        return sigma2s

    def neg_log_likelihood(self, params, returns):
        """return negative log likelihood"""
        T = len(returns)

        sigma2s = self.arch(params, returns)

        llh = -(T - self.p + 1) / 2 * np.log(2 * np.pi)
        for t in range(self.p, T):
            llh -= 1 / 2 * returns[t] ** 2 / sigma2s[t]
            llh -= 1 / 2 * np.log(sigma2s[t])

        return -llh

    def fit(self, returns):
        """return scipy optimize result"""
        params = np.random.uniform(0, 1, self.p+1)
        bounds = [(0, None)] + [(0, 1) for i in range(self.p)]
        returns = np.array(returns)
        self.res = minimize(self.neg_log_likelihood, params, args=(returns), bounds=bounds)
        return self.res
    
    
class GarchModel:
    def __init__(self, p, q):
        self.p = p
        self.q = q
        
    def garch(self, params, returns):
        """return a series of sigma2"""
        omega = params[0]
        alphas = params[1:]
                
        T = len(returns)
        p = len(alphas)
        
        return2s = returns ** 2
        
        sigma2s = np.zeros(T)
        sigma2s[0:p] = np.var(returns)
        for t in range(p, T):
            sigma2s[t] = omega + alphas @ return2s[t-p:t]
        
        if np.any(sigma2s<=0):
            raise ValueError('sigma_2_t <= 0')
        return sigma2s

    def neg_log_likelihood(self, params, returns):
        """return negative log likelihood"""
        T = len(returns)
        p = len(params) - 1

        sigma2s = self.arch(params, returns)

        llh = -(T - p + 1) / 2 * np.log(2 * np.pi)
        for t in range(p, T):
            llh -= 1 / 2 * returns[t] ** 2 / sigma2s[t]
            llh -= 1 / 2 * np.log(sigma2s[t])

        return -llh

    def fit(self, returns):
        """return scipy optimize result"""
        params = np.random.uniform(0, 1, self.p+1)
        bounds = [(0, None)] + [(0, 1) for i in range(self.p)]
        returns = np.array(returns)
        self.res = minimize(self.neg_log_likelihood, params, args=(returns), bounds=bounds)
        return self.res


class CaviarModel:
    def __init__(self, quantile, model='asymmetric'):
        """
        :param: quantile (float): between 0 and 1 exclusively
        :param: model (str): type of CAViaR model
        :param: G (int): some positive number for the smoothen version of indicator function
        """
        if quantile > 0 and quantile < 1:
            self.quantile = quantile
        else:
            raise ValueError('Quantile must be within 0, 1 exclusively.')
        
        if model in ['adaptive', 'symmetric', 'asymmetric', 'igarch']:
            self.model = model
        else:
            raise ValueError('Model must be one of {"adaptive", "symmetric", "asymmetric", "igarch"}')
        
    def adaptive(self, returns, betas, G=10):
        """
        :param: returns (array): a series of daily returns
        :param: betas (array): a series of coefficients
        :param: G (int): some positive number for the smoothen version of indicator function
        :returns: VaR
        """
        b1 = betas[0]
        sigmas = np.zeros_like(returns)
        sigmas[0] = self.get_empirical_quantile(returns) # np.std(returns) # im not sure
        for t in range(1, len(sigmas)):
            sigmas[t] = sigmas[t-1] + b1 * (1 / (1 + np.exp(G * (returns[t-1] - sigmas[t-1]))) - self.quantile)
        return sigmas
        
    def symmetric_abs_val(self, returns, betas):
        """
        :param: returns (array): a series of daily returns
        :param: betas (array): a series of coefficients
        :returns: VaR
        """
        b1, b2, b3 = betas
        sigmas = np.zeros_like(returns)
        sigmas[0] = self.get_empirical_quantile(returns) # np.std(returns[:300])
        for t in range(1, len(sigmas)):
            sigmas[t] = b1 + b2 * sigmas[t-1] + b3 * abs(returns[t-1])
        return sigmas

    def asymmetric_slope(self, returns, betas):
        """
        :param: returns (array): a series of daily returns
        :param: betas (array): a series of coefficients
        :returns: VaR
        """
        b1, b2, b3, b4 = betas
        sigmas = np.zeros_like(returns)
        sigmas[0] = self.get_empirical_quantile(returns) # np.std(returns) # im not sure
        for t in range(1, len(sigmas)):
            sigmas[t] = b1 + b2 * sigmas[t-1] + max(b3 * returns[t-1], 0) + b4 * min(b3 * returns[t-1], 0)
        return sigmas

    def igarch(self, returns, betas):
        """
        Is there a need to constrain b2 + b3 == 1?
        :param: returns (array): a series of daily returns
        :param: betas (array): a series of coefficients
        :returns: VaR
        """
        b1, b2, b3 = betas
        sigmas = np.zeros_like(returns)
        sigmas[0] = self.get_empirical_quantile(returns)
        for t in range(1, len(sigmas)):
            sigmas[t] = (b1 + b2 * sigmas[t-1]**2 + b3 * returns[t-1]**2)**0.5
        return sigmas

    def neg_log_likelihood(self, params, returns, caviar_func):
        """
        :param: returns (array): a series of daily returns
        :param: betas (array): a series of coefficients
        :param: caviar_func (callable function): a CAViaR function that returns VaR
        :returns: negative log likelihood
        """
        T = len(returns)
        
        tau = params[0]
        betas = params[1:]

        sigmas = caviar_func(returns, betas)

        llh = (1 - T) * np.log(tau)
        for t in range(1, T):
            llh -= 1 / tau * max((self.quantile - 1) * (returns[t] - sigmas[t]),
                                 self.quantile * (returns[t] - sigmas[t]))

        return -llh
    
    def initiate_params(self):
        """
        generate the initial estimate of tau and betas
        
        for the bounds and constraints:
        The first and third of these respond symmetrically to past returns,
        whereas the second allows the response to positive and
        negative returns to be different. All three are mean-reverting
        in the sense that the coefficient on the lagged VaR is not constrained
        to be 1.
        """
        # symmetric and igarch: 3 betas; asymmetric: 4 betas; adaptive: 1 beta
        num_params = 3 if self.model in ['symmetric', 'igarch'] else \
                     4 if self.model == 'asymmetric' else 1
        
        # tau, intercept, coefssss
        bounds = [(1e-4, None), (None, None)] + [(0, 1) for i in range(num_params - 1)]
        return np.random.uniform(0, 1, 1 + num_params), bounds
    
    def get_empirical_quantile(self, returns, until_first=300):
        return np.quantile(returns[:until_first], self.quantile)
    
    def fit(self, returns):
        """
        :param: returns (array): a series of daily returns
        :returns: scipy optimize result
        """
        params, bounds = self.initiate_params()
        returns = np.array(returns)
        self.res = minimize(self.neg_log_likelihood, params,
                            args=(returns, self.asymmetric_slope), bounds=bounds)
        return self.res