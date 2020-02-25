import pandas as pd
import numpy as np
import datetime as dt
from util import get_data, plot_data
import scipy.optimize as spo
from tabulate import tabulate


def optimize_portfolio(sd=dt.datetime(2008, 1, 1),
                       ed=dt.datetime(2009, 1, 1),
                       syms=None,
                       gen_plot=False):
    """ Find allocation, and optimized portfolio stats """
    # Read in adjusted closing prices for given symbols, date range
    if syms is None:
        syms = ['GOOG', 'AAPL', 'GLD', 'XOM']
    dates = pd.date_range(sd, ed)
    # automatically adds SPY
    prices_all = get_data(syms, dates)
    # only portfolio symbols
    prices = prices_all[syms]
    # only SPY, for comparison later
    prices_SPY = prices_all['SPY']

    # find the allocations for the optimal portfolio
    num_of_stocks = len(syms)
    allocs = find_optimal_allocations(prices, num_of_stocks)

    # Get Cumulative Return, Average Daily Return, Volatility (stdev of daily returns)
    # Sharpe Ratio and End value
    cr, adr, sddr, sr, ev = assess_portfolio(sd=sd, ed=ed,
                                             syms=syms,
                                             allocs=allocs,
                                             gen_plot=False)

    # Get daily portfolio value
    port_val = get_portfolio_value(normalize_df(prices),
                                   allocs,
                                   start_val=1000000)

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        df_temp = pd.concat([port_val, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
        plot_normalized_data(df_temp, title="Daily Portolio Value and SPY")

    return allocs, cr, adr, sddr, sr


def find_optimal_allocations(prices, num_of_stocks):
    """ Use Scipy to find optimized portfolio allocation """

    # bounds for all stock allocation would be 0 - 1
    bounds = [(0, 1)] * num_of_stocks
    # allocation must sum to 1
    constraints = ({'type': 'eq', 'fun': lambda inputs: 1.0 - np.sum(inputs)})
    # intial guess balanced alloc
    x0 = np.ones(num_of_stocks) * (1. / num_of_stocks)
    # Use negative Sharpe ratio to optimize (minimize)
    result = spo.minimize(negative_sharpe, x0, prices, method='SLSQP',
                          bounds=bounds, constraints=constraints)
    return result.x


def negative_sharpe(X, prices, samples_per_year=252., risk_free_rate=0., sv=1000000):
    """ Return portfolio's negative sharpe ratio """
    # Get daily portfolio value
    port_val = get_portfolio_value(normalize_df(prices), X, sv)
    # Get portfolio statistics (note: std_daily_ret = volatility)
    daily_rets = compute_daily_returns(port_val)[1:]
    # Get portfolio stats
    cr, adr, sddr, sr = get_portfolio_stats(port_val, daily_rets,
                                            samples_per_year, risk_free_rate)
    return -sr


def compute_daily_returns(df):
    """Compute and return the daily return values."""
    return ((df / df.shift(1)) - 1).fillna(0)


def get_portfolio_value(prices, allocs, start_val):
    """ Compute daily portfolio value given stock prices, allocations and starting value.

        Return: pandas Series or DataFrame (with a single column)
    """
    return (prices * allocs).sum(1) * start_val


def get_portfolio_stats(port_val, daily_rf, samples_per_year, risk_free_rate):
    """ Calculate statistics on daily portfolio value, given daily risk-free rate and data sampling frequency.

        Return: a tuple consisting of the following statistics (in order): cumulative return, average daily return, standard deviation of daily return, Sharpe ratio
        Note: The return statement provided ensures this order.
    """

    # Get portfolio statistics (note: std_daily_ret = volatility)
    daily_rf = compute_daily_returns(port_val)[1:]
    cr = get_cumulative_return(port_val)
    adr = daily_rf.mean()
    sddr = daily_rf.std()

    # Sharpe ratio = K * (Return of Portfolio - Risk-free Rate) / SD of portfolio's excess return
    # Daily sampling K = sqrt(252.)
    sr = np.sqrt(samples_per_year) * (adr - risk_free_rate) / sddr

    return cr, adr, sddr, sr


def normalize_df(port_val):
    """ Return normalized dataframe starting with 1.0 """
    return port_val / port_val.iloc[0]


def get_cumulative_return(df):
    """ Get Cumulative return of Portfolio"""
    return df[-1] / df[0] - 1.


def plot_normalized_data(df, title="Portfolio vs SPY", xlabel="Date", ylabel="Normalized Return"):
    """ Plot normalized return """
    plot_data(normalize_df(df), title, xlabel, ylabel)


def assess_portfolio(sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 1, 1),
                     syms=None,
                     allocs=None,
                     sv=1000000,
                     rfr=0.0,
                     sf=252.0,
                     gen_plot=False):
    # Read in adjusted closing prices for given symbols, date range
    if syms is None:
        syms = ['GOOG', 'AAPL', 'GLD', 'XOM']
    if allocs is None:
        allocs = [0.1, 0.2, 0.3, 0.4]
    # get date range
    dates = pd.date_range(sd, ed)
    # print dates
    prices_all = get_data(syms, dates)  # automatically adds SPY
    # print prices_all.head()
    prices = prices_all[syms]  # only portfolio symbols
    # print prices.head()
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # Get daily portfolio value
    port_val = get_portfolio_value(normalize_df(prices), allocs, sv)

    # Get portfolio statistics (note: std_daily_ret = volatility)
    daily_rets = compute_daily_returns(port_val)[1:]

    cr, adr, sddr, sr = get_portfolio_stats(port_val, daily_rf=daily_rets,
                                            samples_per_year=sf, risk_free_rate=rfr)

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        df_temp = pd.concat([port_val, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
        plot_normalized_data(df_temp)

    # Add code here to properly compute end value
    ev = port_val[-1]

    return cr, adr, sddr, sr, ev


def test_code():
    # This function WILL NOT be called by the auto grader
    # Do not assume that any variables defined here are available to your function/code
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!

    start_date = dt.datetime(2008, 6, 1)
    end_date = dt.datetime(2009, 6, 1)
    symbols = ['IBM', 'X', 'GLD', 'JPM']

    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd=start_date,
                                                        ed=end_date,
                                                        syms=symbols,
                                                        gen_plot=False)

    output = tabulate([['Start Date', start_date],
                       ['End Date', end_date],
                       ['Symbols', symbols],
                       ['Allocations', allocations],
                       ['Sharpe Ratio', sr],
                       ['Volatility', sddr],
                       ['Average Daily Return', adr],
                       ['Cumulative Return', cr]],
                      headers=['Name', 'Value'])

    print(output)

    # # Print statistics
    # print("Start Date:", start_date)
    # print("End Date:", end_date)
    # print("Symbols:", symbols)
    # print("Allocations:", allocations)
    # print("Sharpe Ratio:", sr)
    # print("Volatility (stdev of daily returns):", sddr)
    # print("Average Daily Return:", adr)
    # print("Cumulative Return:", cr)


if __name__ == "__main__":
    test_code()
