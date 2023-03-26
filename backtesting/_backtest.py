import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def backtest(returns, low_open_log_difference, VaRs, quantile, ntl=100, penalty=0.002):
    """
    replace the return over q%-quantile VaR from day 1 to day T
    
    logic behind: stong hold if nothing happens (no violation)
    once log(low/open) touches the VaR => sell immediately
    and you buy it back at the closing price when the market is almost closed.
    
    :param: returns (array-like): returns from day 0 to day T
    :param: low_open_log_difference (array-like): log(Low) - log(Open) from day 0 to day T
    :param: VaRs (CaviarModel object): VaR for day 1 to day T+1: VaR_t = f(info at time t-1)
            as the returns are from 0 to T => VaR are for 1 to T+1
    :param: ntl (float): positive notional. Default is 100.
    :param: penalty (float): Each default transaction cost 0.2%, i.e. 0.002
    """
    # must be positive
    if ntl < 0:
        ntl = 100
    
    returns = pd.Series(returns)
    date = returns.index
    
    backtest_df = pd.DataFrame({
        'ret': returns[1:] / 100,
        'low/open': low_open_log_difference[1:],
        'VaR': VaRs[:-1] / 100
    })
    
    original = ntl * (1 + backtest_df.ret).cumprod()
    
    new = (1 + np.maximum(backtest_df['ret'], backtest_df['VaR']))
    new = new * ((backtest_df['low/open'] < backtest_df['VaR']) * -penalty + 1)
    new = ntl * new.cumprod()
    
    plt.figure(figsize=(8, 6))
    plt.plot(date[1:], original, label='original')
    plt.plot(date[1:], new, label=f'with {int(quantile*100)}% risk control')
    plt.title('Cummulative Return Plot')
    plt.legend()
    plt.yscale('log')
    plt.show()