# Author: Lee Yat Shun, Jasper
# Copyright (c) 2023 Lee Yat Shun, Jasper. All rights reserved.

import numpy as np
import pandas as pd
from scipy.stats import chi2, binom_test, binom, norm


hit_rate = lambda ret, var: np.mean(ret < var)

def binomial_test(returns, VaRs, quantile):
    """
    null hypothesis: the probability of success/failure in a Bernoulli experiment is p.
    
    https://ww2.mathworks.cn/help/risk/overview-of-var-backtesting.html
    
    :params: returns (array):
    :params: VaRs (array):
    :params: quantile (float):
    :returns: two-sided binomial test
    """
    k = np.sum(returns < VaRs) # number of failures
    n = len(returns) # num of total observations
    return binom_test(k, n, p=quantile)


def traffic_light_test(returns, VaRs, quantile, num_obs=250, baseline=3):
    """
    For notation: Pr = Pr(X<=k|quantile)
    
    if Pr <= 0.95: return green (accurate)
    else if Pr <= 0.99: return yellow (between accurate and inaccurate)
    else: return red (totally inaccurate)
    
    :param: returns (array-like):
    :param: VaRs (array-like):
    :param: quantile (float):
    :param: num_obs (int): last Default is 250.
    :param: baseline (int): Default is 3.
    :returns: tl (string): either green, yellow, or red
    :returns: p (float): between 0 to 1.
    :returns: scale_factor (float): between 0 to 1.
    """
    num_violations = np.sum(returns[-num_obs:] < VaRs[-num_obs:])
    
    z_assumed = norm.ppf(quantile)
    z_observed = norm.ppf(num_violations / num_obs)
    p = binom.cdf(num_violations, num_obs, quantile)
    
    tl = 'green' if p <= 0.95 else 'yellow' if p <= 0.9999 else 'red'
    scale_factor = baseline * (z_assumed/z_observed - 1) if tl == 'yellow' else 0 if tl == 'green' else 1
    
    return tl, p, scale_factor


def kupiec_pof_test(returns, VaRs, quantile):
    """
    null hypothesis: the observed failure rate is equal to
                     the failure rate suggested by the confidence interval.
                     
    This statistic is asymptotically distributed as a chi-square variable with 1 degree of freedom.
    https://ww2.mathworks.cn/help/risk/overview-of-var-backtesting.html
    
    :param: returns (array-like):
    :param: VaRs (array-like):
    :param: quantile (float):
    :returns: pvalue
    """
    # p = 1 - VaR level, where (quantile = 0.05) == (VaR_level = 0.95)
    p = quantile
    x = np.array(returns < VaRs).sum()
    N = len(returns)
    
    numerator = (1 - p) ** (N-x) * p ** x
    denominator = (1 - x / N) ** (N-x) * (x / N) ** x
    LR_POF = -2 * np.log(numerator / denominator)
    return chi2.cdf(LR_POF, df=1)
    
    
def christoffersen_test(returns, VaRs):
    """
    independence test by Christoffersen, 1998
    null hypothesis: the observations are independent of each other
    
    This statistic is asymptotically distributed as a chi-square variable with 1 degree of freedom.
    https://ww2.mathworks.cn/help/risk/overview-of-var-backtesting.html
    
    :param: returns (array-like):
    :param: VaRs (array-like):
    :param: quantile (float):
    :returns: pvalue
    """
    violations = np.array(returns < VaRs) * 1
    diff = violations[:-1] - violations[1:] # current one - followed by
    
    # Number of periods with no failures followed by a period with failures.
    n01 = (diff==-1).sum() 
    # Number of periods with failures followed by a period with no failures.
    n10 = (diff==1).sum()
    # Number of periods with no failures followed by a period with no failures.
    n00 = (violations[1:][diff==0]==0).sum()
    # Number of periods with failures followed by a period with failures.
    n11 = (violations[1:][diff==0]==1).sum()
    
    pi0 = n01 / (n00 + n01) # pr(fail given no fail at t-1)
    pi1 = n11 / (n10 + n11) # pr(fail given fail at t-1)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11) # pr(fail on t)
    
    numerator = (1 - pi) ** (n00 + n10) * pi ** (n01 + n11)
    denominator = (1 - pi0) ** n00 * pi0 ** n01 * (1 - pi1) ** n10 * pi1 ** n11
    LR_CCI = -2 * np.log(numerator / denominator)
    return chi2.cdf(LR_CCI, df=1)
