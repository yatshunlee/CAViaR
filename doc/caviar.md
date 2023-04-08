## Summary
In the paper "CAViaR: Conditional Autoregressive Value at Risk by Regression Quantiles" by Robert F. Engle and Simone Manganelli (2004), the authors propose a new method for estimating and evaluating Value-at-Risk (VaR) forecasts. The main idea is to model the quantiles of future returns directly, rather than modeling the entire distribution of returns. The model they propose is called the Conditional Autoregressive Value at Risk (CAViaR) model.

This library `caviar` is designed based on the paper and our additional research on the parameter setting and optimization method.

## 4 types of CAViaR model in Engle & Manganelli, 2004
### Adaptive:
$$f_{t}(\beta_{1}) = f_{t-1}(\beta_{1}) + \beta_{1} \cdot ([1 + \exp(G[y_{t-1} - f_{t-1}(\beta_{1})])]^{-1} - \theta )$$

### Symmetric absolute value:
$$f_t(\beta) = \beta_{1} + \beta_{2} f_{t-1}(\beta) + \beta_{3} |y_{t-1}|$$

### Asymmetric slope:
$$f_t(\beta) = \beta_{1} + \beta_{2} f_{t-1}(\beta) + \beta_{3} \cdot max(y_{t-1}, 0) + \beta_{4} \cdot [-min(y_{t-1}, 0)]$$

### IGARCH(1, 1):
$$f_t(\beta) = \sqrt{\beta_{1} + \beta_{2} f_{t-1}^2(\beta) + \beta_{3} y_{t-1}^2}$$

## Method
### RQ Criterion Minimization (RQ):
$$min_\beta T^{-1} \sum_{t=1}^T [ \theta - I(y_t < f_t(\beta)) ] [y_t - f_t(\beta)]$$

where:
- $y_t$ is the return at period t (author using log return * 100)
- $f_t(\beta)$ is the predicted VaR at period t
- $\theta$ is the quantile level, which ranges from 0 to 1

### Minimizing Negative Log-Likelihood (MLE):
$$min_{\beta, \tau} Tlog{\tau} + {\tau}^{-1}\sum_{t=1}^T [ \theta - I(y_t < f_t(\beta)) ] [y_t - f_t(\beta)]$$

where:
- $y_t$ is the return at period t, assumming $y_t$ follows Asymmetric Skewed Laplace Distribution with $\tau$ > 0
- $f_t(\beta)$ is the predicted VaR at period t
- $\theta$ is the quantile level, which ranges from 0 to 1

### Optimization Method
We optimized the problem differently. Instead of best start: picking m best $\beta$ from n random start, we use random start method, m = n = 1, since we have found that best start and random start don't show a huge difference on the loss value and hit rate (in and out of samples) in the experiment of testing their repeatabilities.

Instead of usibg simplex algorithm followed by quasi-newton method, we have used L-BFGS-B to optimize the problems.

## Example
```
# firstly initialize the in-sample and out-of-sample returns
in_samples = some_returns[in] * 100
out_samples = some_returns[out] * 100

# initialize the parameters by choosing from these
# q = 0 to 1 exclusively
# model is from {'adaptive', 'asymmetric', 'symmetric', 'igarch'}
# method is from {'RQ', 'mle'}

# declare a model instance
caviar_model = CaviarModel(q, model, method)
# fit the beta
caviar_model.fit(in_samples)
# print the statistic of beta
print(caviar_model.beta_summary())

# predict the fittedvalues
in_VaR_predicted = caviar_model.predict(in_samples, 'in')
out_VaR_predicted = caviar_model.predict(out_samples, 'out')

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
