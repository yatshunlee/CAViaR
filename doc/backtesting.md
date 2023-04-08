## Backtesting
This library `backtesting` is specifically designed for using volatility model to set stop loss for a portfolio and values its annualized return, cummulative return and maximium drawdown.

### Logic Behind: stong hold if nothing happens (no violation)
- if log(low/open) touches/exceeds the VaR => sell immediately and you buy it back at the closing price.
- else => do nothing

### How to use? (One example)
```
# returns = return in (%), where it's 100x log return, i.e. original: [0.01] => 100-times: [1.0]
# low_open_log_difference = log(low/open) at the same day (not required to be 100x)
# VaRs = stop loss (%), also 100 times
# ntl: initial capital
# penalty: 0.2% as transaction cost
stat = backtest(returns, low_open_log_difference, VaRs, ntl=100, penalty=0.002)

pd.DataFrame([stat], columns=[
    'annualized return (with stop loss)',
    'cummulative return (with stop loss)',
    'max. drawdown (with stop loss)',
    'annualized return (original)',
    'cummulative return (original)',
    'max. drawdown (original)',
])
```
