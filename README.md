# CAViaR-Project
for estimating Conditional Autoregressive Value at Risk (CAViaR) models.
## 4 types of CAViaR model in Engle & Manganelli, 2004
### Adaptive:
$f_{t}(\beta_{1}) = f_{t-1}(\beta_{1}) + \beta_{1} \cdot ([1 + \exp(G[y_{t-1} - f_{t-1}(\beta_{1})])]^{-1} - \theta )$

### Symmetric absolute value:
$f_t(\beta) = \beta_{1} + \beta_{2} f_{t-1}(\beta) + \beta_{3} |y_{t-1}|$

### Asymmetric slope:
$f_t(\beta) = \beta_{1} + \beta_{2} f_{t-1}(\beta) + \beta_{3} \cdot max(y_{t-1}, 0) + \beta_{4} \cdot [-min(y_{t-1}, 0)]$

### IGARCH(1, 1):
$f_t(\beta) = \sqrt{\beta_{1} + \beta_{2} f_{t-1}^2(\beta) + \beta_{3} y_{t-1}^2}$

## Doc
### Example of fitting a CAViaR model with asymmetric slope
```
log_returns = ... # some returns in array, series, list, ...
in_samples = log_returns [2000:-2000] * 100
out_of_samples = log_returns [-2000:] * 100

# initiate parameter
quantile = 0.05 # 5% VAR
model = 'asymmetric' # asymmetric slope caviar function

# fitting process
caviar_model = CaviarModel(quantile, model='asymmetric', method='numeric')
caviar_model.fit(in_samples)

# you can see the statistics of beta
print('S.E. of beta:', caviar_model.beta_standard_errors)
print('pval of beta:', caviar_model.beta_pvals)

# backtesting
# in sample
caviar_model.plot_caviar(in_samples)

# out of sample
caviar_model.plot_caviar(out_of_samples)

# plot the news impact curve
caviar_model.plot_news_impact_curve(VaR=-1.645)

# DQ test
print(caviar_model.dq_test(in_samples, 'in'))
print(caviar_model.dq_test(out_of_samples, 'out'))

# Forecast day t+1
VaR = caviar_model.predict(out_of_samples)
VaR_forecast = - caviar_model.forecast(out_of_samples[-1], VaR[-1]) # since it's reported postively
print(VaR_forecast) 
```
