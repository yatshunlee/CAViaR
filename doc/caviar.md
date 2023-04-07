## Summary
In the paper "CAViaR: Conditional Autoregressive Value at Risk by Regression Quantiles" by Robert F. Engle and Simone Manganelli (2004), the authors propose a new method for estimating and evaluating Value-at-Risk (VaR) forecasts. The main idea is to model the quantiles of future returns directly, rather than modeling the entire distribution of returns. The model they propose is called the Conditional Autoregressive Value at Risk (CAViaR) model.

## 4 types of CAViaR model in Engle & Manganelli, 2004
### Adaptive:
$f_{t}(\beta_{1}) = f_{t-1}(\beta_{1}) + \beta_{1} \cdot ([1 + \exp(G[y_{t-1} - f_{t-1}(\beta_{1})])]^{-1} - \theta )$

### Symmetric absolute value:
$f_t(\beta) = \beta_{1} + \beta_{2} f_{t-1}(\beta) + \beta_{3} |y_{t-1}|$

### Asymmetric slope:
$f_t(\beta) = \beta_{1} + \beta_{2} f_{t-1}(\beta) + \beta_{3} \cdot max(y_{t-1}, 0) + \beta_{4} \cdot [-min(y_{t-1}, 0)]$

### IGARCH(1, 1):
$f_t(\beta) = \sqrt{\beta_{1} + \beta_{2} f_{t-1}^2(\beta) + \beta_{3} y_{t-1}^2}$


## Example
```
# firstly initialize the in-sample and out-of-sample returns
in_samples = some_returns[in] * 100
out_samples = some_returns[out] * 100

# initialize the parameters
# q = 0 to 1 exclusively
# model = {'adaptive', 'asymmetric', 'symmetric', 'igarch'}
# method = {'numeric', 'mle'}

# declare a model instance
caviar_model = CaviarModel(q, model, method)
# fit the beta
caviar_model.fit(in_samples)
# print the statistic of beta
print(caviar_model.beta_summary())

# predict the fittedvalues
in_VaR_predicted = caviar_model.predict(in_samples, caviar_model.VaR0_in)
out_VaR_predicted = caviar_model.predict(out_samples, caviar_model.VaR0_out)

# get the predicted values
in_VaR_fitted, in_VaR_forecast = in_VaR_predicted[:-1], in_VaR_predicted[-1]
out_VaR_fitted, out_VaR_forecast = out_VaR_predicted[:-1], out_VaR_predicted[-1]

# DQ Test [OPTIONAL]
# the null hypothesis of the Dynamic Quantile Test used in the CAViaR paper is that
# the CAViaR model provides accurate VaR forecasts for a given confidence level
print(caviar_model.dq_test(in_samples, 'in'))
print(caviar_model.dq_test(out_samples, 'out'))
```

Notice that since the model is a time series model, when you want to perform out of sample prediction, the in-sample and out-of-sample data must be consecutive. If the in-sample are from 0, 1, ..., T, then the out-of-sample must be starting from T+1, T+2, ...
