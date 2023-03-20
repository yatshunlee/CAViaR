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
```
class CaViarModel(quantile, model, method='numeric', G=10, tol=1e-10)

parameters:
- quantile (float): Quantile value between 0 and 1 exclusively. Default is 0.05
- model (str): Type of CAViaR model. Model must be one of {"adaptive", "symmetric", "asymmetric", "igarch"}
               Default is "asymmetric", i.e., asymmetric slope.
- method (str): Estimation method. Must be one of {"numeric (Engle & Manganelli, 2004)",
                "mle (Maximum Likelihood Estimation)"}.
                Default is "numeric".
- G (int): Smoothen version of the indicator function. Some positive number. Default is 10.
- tol (float): Tolerance level for optimization. Default is 1e-10.

attributes:
- beta (np.array): estimated beta of caviar function (available after fitting)
- caviar (callable): chosen caviar function (available after fitting)
```

### Example of fitting a CAViaR model with asymmetric slope
```
log_returns = ... # some returns in array, series, list, ...

# initiate parameter
quantile = 0.05 # 5% VAR
mod = 'asymmetric' # asymmetric slope caviar function

# fitting process
caviar_model = CaViarModel(quantile, mod)
caviar_model.fit(log_returns)

# result
beta = caviar_model.beta
vars = caviar_model.caviar(log_returns, betas, quantile)
```
