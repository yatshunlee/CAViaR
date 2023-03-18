import numpy as np
from scipy.optimize import minimize


class Caviar:
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

    def get_empirical_quantile(self, returns, until_first=300):
        return np.quantile(returns[:until_first], self.quantile)

    def adaptive(self, returns, betas):
        """
        :param: returns (array): a series of daily returns
        :param: betas (array): a series of coefficients
        :param: G (int): some positive number for the smoothen version of indicator function
        :returns: VaR
        """
        b1 = betas[0]
        sigmas = np.zeros_like(returns)
        sigmas[0] = self.get_empirical_quantile(returns)  # np.std(returns) # im not sure
        for t in range(1, len(sigmas)):
            sigmas[t] = sigmas[t - 1] + b1 * (
                        1 / (1 + np.exp(self.G * (returns[t - 1] - sigmas[t - 1]))) - self.quantile)
        return sigmas

    def symmetric_abs_val(self, returns, betas):
        """
        :param: returns (array): a series of daily returns
        :param: betas (array): a series of coefficients
        :returns: VaR
        """
        b1, b2, b3 = betas
        sigmas = np.zeros_like(returns)
        sigmas[0] = self.get_empirical_quantile(returns)  # np.std(returns[:300])
        for t in range(1, len(sigmas)):
            sigmas[t] = b1 + b2 * sigmas[t - 1] + b3 * abs(returns[t - 1])
        return sigmas

    def asymmetric_slope(self, returns, betas):
        """
        :param: returns (array): a series of daily returns
        :param: betas (array): a series of coefficients
        :returns: VaR
        """
        b1, b2, b3, b4 = betas
        sigmas = np.zeros_like(returns)
        sigmas[0] = self.get_empirical_quantile(returns)  # np.std(returns) # im not sure
        for t in range(1, len(sigmas)):
            sigmas[t] = b1 + b2 * sigmas[t - 1] + max(b3 * returns[t - 1], 0) + b4 * min(b3 * returns[t - 1], 0)
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
            sigmas[t] = (b1 + b2 * sigmas[t - 1] ** 2 + b3 * returns[t - 1] ** 2) ** 0.5
        return sigmas

    def obj(self, betas, returns, caviar):
        """
        :param: betas (array-like): parameters of CAVIAR function
        :param: returns (array-like): a series of returns
        :param: caviar (callable function): a CAVIAR function
        :return: quantile regression loss
        """
        sigmas = caviar(returns, betas)
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
            self.model = self.adaptive
            self.p = 1
        elif self.model == 'symmetric':
            self.model = self.symmetric_abs_val
            self.p = 3
        elif self.model == 'asymmetric':
            self.model = self.asymmetric_slope
            self.p = 4
        else:  #IGARCH
            self.model = self.igarch
            self.p = 3

        if self.method == 'numeric':
            return self.numeric_fit(returns)

        elif self.method == 'mle':
            return self.mle_fit(returns)

    def numeric_fit(self, returns):
        """
        following Engle & Manganelli (2004) approach
        """
        initial_betas = self.initialize_betas(returns)
        result = []

        for initial_beta in initial_betas:
            beta, loss = self.optimize(initial_beta, returns)
            result.append(
                {
                    'beta': beta,
                    'loss': loss
                }
            )

        result = sorted(result, key=lambda x: x['loss'])

        self.beta = result['beta']

    def initialize_betas(self, returns):
        """
        :param: returns (np.array): a series of returns

        n (int): We generated n vectors using a uniform random number generator between 0 and 1.
        m (int): We computed the regression quantile (RQ) function for each of these vectors and
                 selected the m vectors that produced the lowest RQ criterion as initial values
                 for the optimization routine
        """
        if self.model == 'adaptive':
            n = 10 ** 4
            m = 5
        elif self.model == 'symmetric':
            n = 10 ** 4
            m = 10
        elif self.model == 'asymmetric':
            n = 10 ** 5
            m = 15
        else:  # IGARCH
            n = 10 ** 4
            m = 10

        random_betas = np.random.uniform(0, 1, (n, p))

        best_initial_betas = []

        # if have time, can try heap instead of sorted list
        for i in range(n):
            loss = self.obj(random_betas[i], returns, self.model)

            best_initial_betas.append(
                {'loss': loss, 'beta': random_betas[i]}
            )

            best_initial_betas = sorted(best_initial_betas, key=lambda x: x['loss'])

            if len(best_initial_betas) == m:
                best_initial_betas.pop()

        return best_initial_betas

    def optimize(self, initial_beta, returns):
        beta_new = initial_beta['beta']
        current_loss = initial_beta['loss']

        while True:
            # Minimize the function using the Nelder-Mead algorithm
            res = minimize(self.obj, beta_new, args=(returns, self.model), method='Nelder-Mead')
            beta_new = res.x

            loss = res.fun

            if current_loss - loss < self.tol:
                break
            else:
                current_loss = loss

            # Minimize the function using the BFGS algorithm
            res = minimize(self.obj, beta_new, args=(returns, self.model), method='BFGS')
            beta_new = res.x

            loss = res.fun

            if current_loss - loss < self.tol:
                break
            else:
                current_loss = loss

        return beta_new, loss

    def mle_fit(self, returns):
        """
        :param: returns (array): a series of daily returns
        :returns: scipy optimize result
        """
        params, bounds = self.initiate_params
        result = minimize(self.neg_log_likelihood, params,
                          args=(returns, self.model), bounds=bounds)

        self.beta = result.x

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
        # tau, intercept, coefficients
        bounds = [(1e-4, None), (None, None)] + [(-1, 1) for _ in range(self.p - 1)]
        return np.random.uniform(0, 1, 1 + self.p), bounds

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

