## Summary
The objective is to determine whether the VaR model is reliable and accurate in predicting the actual losses in different market conditions, by statistically measuring comparing the actual losses of an investment or portfolio with the predicted VaR estimates.

This library `var_tests` consists of:
- Binomial test
- Traffic light test
- Kupiec’s POF test
- Christoffersen’s test
- Dynamic Quantile test

We referred to the "Overview of VAR Backtesting" page in the Matlab official website to build the statistical test.

## How to use
```
# For binomial test: 
# Compare the observed number of exceptions with the expected one
# Rejected when there are too few/many violations
pval = binomial_test(returns, VaRs, quantile)

# For traffic light test: 
# Only too many exceptions lead to rejection
# zone = {'green', 'yellow', 'red'}
zone, p, scale_factor = binomial_test(returns, VaRs, quantile)

# For Kupiec’s POF test: 
# If the data suggests that the probability of exceptions is different than p, the VaR model is rejected.
pval = kupiec_pof_test(returns, VaRs, quantile)

# For Christoffersen’s test: 
# It measures the dependency between consecutive days only.
pval = christoffersen_test(returns, VaRs)

# For DQ test: 
# It measures the unbiasedness and independence of VaR estimate
pval = dq_test(returns, VaRs, quantile)
```

## Reference
- MathWorks. (n.d.). Overview of VAR Backtesting. Retrieved March 31, 2023, from https://ww2.mathworks.cn/help/risk/overview-of-var-backtesting.html
