import numpy as np


def get_empirical_quantile(returns, quantile, until_first=300):
    return np.quantile(returns[:until_first], quantile)


def adaptive(returns, betas, quantile, G=10):
    """
    :param: returns (array): a series of daily returns
    :param: betas (array): a series of coefficients
    :param: G (int): some positive number for the smoothen version of indicator function
    :returns: VaR
    """
    b1 = betas[0]
    sigmas = np.zeros_like(returns)
    sigmas[0] = get_empirical_quantile(returns)
    for t in range(1, len(sigmas)):
        sigmas[t] = sigmas[t - 1] + b1 * (
                1 / (1 + np.exp(G * (returns[t - 1] - sigmas[t - 1]))) - quantile)
    return sigmas


def symmetric_abs_val(returns, betas, quantile):
    """
    :param: returns (array): a series of daily returns
    :param: betas (array): a series of coefficients
    :returns: VaR
    """
    b1, b2, b3 = betas
    sigmas = np.zeros_like(returns)
    sigmas[0] = get_empirical_quantile(returns, quantile)
    for t in range(1, len(sigmas)):
        sigmas[t] = b1 + b2 * sigmas[t - 1] + b3 * abs(returns[t - 1])
    return sigmas


def asymmetric_slope(returns, betas, quantile):
    """
    :param: returns (array): a series of daily returns
    :param: betas (array): a series of coefficients
    :returns: VaR
    """
    b1, b2, b3, b4 = betas
    sigmas = np.zeros_like(returns)
    sigmas[0] = get_empirical_quantile(returns, quantile)
    for t in range(1, len(sigmas)):
        sigmas[t] = b1 + b2 * sigmas[t - 1] + max(b3 * returns[t - 1], 0) - b4 * min(b3 * returns[t - 1], 0)
    return sigmas


def igarch(returns, betas, quantile):
    """
    Is there a need to constrain b2 + b3 == 1?
    :param: returns (array): a series of daily returns
    :param: betas (array): a series of coefficients
    :returns: VaR
    """
    b1, b2, b3 = betas
    sigmas = np.zeros_like(returns)
    sigmas[0] = get_empirical_quantile(returns, quantile)
    for t in range(1, len(sigmas)):
        sigmas[t] = (b1 + b2 * sigmas[t - 1] ** 2 + b3 * returns[t - 1] ** 2) ** 0.5
    return sigmas
