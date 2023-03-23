# Author: Lee Yat Shun, Jasper
# Copyright (c) 2023 Lee Yat Shun, Jasper. All rights reserved.

import numpy as np
from scipy.stats import chi2, binom_test


def binomial_test(returns, VaRs, quantile):
    """
    null hypothesis that the probability of success/failure in a Bernoulli experiment is p.
    
    :params: returns (array):
    :params: VaRs (array):
    :params: quantile (float):
    :returns: two-sided binomial test
    """
    k = count_violations(returns, VaRs) # number of failures
    n = len(returns) # num of total observations
    return binom_test(k, n, p=quantile)

def vrate(returns, VaRs):
    """
    :params: returns (array):
    :params: VaRs (array):
    returns: VRate (%): the violation rate
    """
    return count_violations(returns, VaRs)/returns.shape[0]
    
def count_violations(returns, VaRs):
    """
    :params: returns (array):
    :params: VaRs (array):
    :returns: number of violations
    """
    returns = np.array(returns)
    return np.sum(returns < VaRs)