import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def backtest(returns, low_open_log_difference, VaRs, ntl=100, penalty=0.002, ticker=None):
    """
    replace the return over q%-quantile VaR from day 1 to day T
    
    logic behind: stong hold if nothing happens (no violation)
    once log(low/open) touches the VaR => sell immediately
    and you buy it back at the closing price when the market is almost closed.
    
    :param: returns (array-like): returns from day 0 to day T
    :param: low_open_log_difference (array-like): log(Low) - log(Open) from day 0 to day T
    :param: VaRs (CaviarModel object): VaR for day 0 to day T
    :param: ntl (float): positive notional. Default is 100.
    :param: penalty (float): Each default transaction cost 0.2%, i.e. 0.002
    """
    if (len(returns) != len(VaRs) or
        len(returns) != len(low_open_log_difference) or
        len(VaRs) != len(low_open_log_difference)):
        raise Exception('len not align')
    
    # must be positive
    if ntl < 0:
        ntl = 100
    
    returns = pd.Series(returns)
    date = returns.index
    
    backtest_df = pd.DataFrame({
        'ret': returns / 100,
        'low/open': low_open_log_difference,
        'VaR': VaRs / 100
    })
    
    original = ntl * (1 + backtest_df.ret).cumprod()
    
    new = (1 + np.maximum(backtest_df['ret'], backtest_df['VaR']))
    new = new * ((backtest_df['low/open'] < backtest_df['VaR']) * -penalty + 1)
    
    # some statistics
    stat1 = annualized_return(new-1)
    stat2 = cumulative_return(new-1)
    stat3 = maximum_drawdown(new-1)
    
    stat4 = annualized_return(returns/100)
    stat5 = cumulative_return(returns/100)
    stat6 = maximum_drawdown(returns/100)
    new = ntl * new.cumprod()
    
    plt.figure(figsize=(8, 6))
    plt.plot(date, original, label='original')
    plt.plot(date, new, label='with risk control')
    if ticker is None:
        plt.title('Cummulative Return Plot')
    else:
        plt.title(f'Cummulative Return Plot - {ticker}')
    plt.legend()
    plt.yscale('log')
    plt.show()
    
    return stat1, stat2, stat3, stat4, stat5, stat6
    
def annualized_return(returns, periods_per_year=252):
    """
    Calculate the annualized return given a series of investment returns.

    Args:
    returns (pd.Series): A series of investment returns
    periods_per_year (int): The number of periods in a year (default: 252, daily returns)

    Returns:
    float: The annualized return
    """
    returns = returns + 1
    geometric_mean = np.prod(returns) ** (1 / len(returns))
    annualized_return = (geometric_mean ** periods_per_year) - 1
    return annualized_return

def sharpe_ratio(returns, risk_free_rate=0):
    """
    Calculates the Sharpe ratio of a portfolio given its returns and the risk-free rate.

    Parameters:
    returns (numpy.ndarray or list): An array of returns for the portfolio.
    risk_free_rate (float): The risk-free rate of return.

    Returns:
    float: The Sharpe ratio of the portfolio.
    """
    excess_returns = np.array(returns) - risk_free_rate
    mean_excess_returns = np.mean(excess_returns)
    std_excess_returns = np.std(excess_returns)
    sharpe = mean_excess_returns / std_excess_returns
    return sharpe

def cumulative_return(returns):
    """
    Calculate the cumulative return given a series of investment returns.

    Args:
    returns (pd.Series): A series of investment returns

    Returns:
    float: The cumulative return
    """
    returns = returns + 1
    cumulative_return = np.prod(returns) - 1
    return cumulative_return

def maximum_drawdown(returns):
    """
    Calculate the maximum drawdown given a series of investment returns.

    Args:
    returns (pd.Series): A series of investment returns

    Returns:
    float: The maximum drawdown
    """
    cumulative_returns = (returns + 1).cumprod()
    peak_value = cumulative_returns.expanding(min_periods=1).max()
    drawdowns = cumulative_returns / peak_value - 1
    maximum_drawdown = drawdowns.min()
    return maximum_drawdown
