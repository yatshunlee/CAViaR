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

# initiate parameter
quantile = 0.05 # 5% VAR
mod = 'symmetric' # asymmetric slope caviar function

# fitting process
caviar_model = CaviarModel(quantile, mod)
caviar_model.fit(log_returns)

# result
beta = caviar_model.beta
vars = caviar_model.caviar(log_returns*100, beta, quantile)
print("No. samples out of VAR (normalized):", sum(np.where(returns*100 <= vars, 1, 0))/returns.shape[0])
```
