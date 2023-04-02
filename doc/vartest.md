## Summary
To evaluate the performance and accuracy of VaR models, backtesting is often used. Backtesting is a statistical method of comparing the actual losses of an investment or portfolio with the predicted VaR estimates. The objective of backtesting is to determine whether the VaR model is reliable and accurate in predicting the actual losses in different market conditions.

This library consists of:
- Binomial test
- Traffic light test
- Kupiec’s POF test
- Christoffersen’s test

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
```

## Reference
- MathWorks. (n.d.). Overview of VAR Backtesting. Retrieved March 31, 2023, from https://ww2.mathworks.cn/help/risk/overview-of-var-backtesting.html
