"""
Code implementing a ManualStrategy object (your manual strategy).
It should implement testPolicy() which returns a trades data frame (see below).
The main part of this code should call marketsimcode as necessary
    to generate the plots used in the report.

Rules:
* trade only the symbol JPM
- You may use data from other symbols (such as SPY) to inform your strategy.
- The in sample/development period is January 1, 2008 to December 31 2009.
- The out of sample/testing period is January 1, 2010 to December 31 2011.
- Starting cash is $100,000.
- Allowable positions are: 1000 shares long, 1000 shares short, 0 shares.
- Benchmark: The performance of a portfolio starting with $100,000 cash,
    investing in 1000 shares of JPM and holding that position.
- There is no limit on leverage.
- Transaction costs for ManualStrategy: Commission: $9.95, Impact: 0.005.
- Correct trades df format used.

Hints:
* Rule based design:
- Use a cascade of if statements conditioned on the indicators to identify
    whether a LONG condition is met.
- Use a cascade of if statements conditioned on the indicators to identify
    whether a SHORT condition is met.
- The conditions for LONG and SHORT should be mutually exclusive.
- If neither LONG or SHORT is triggered, the result should be DO NOTHING.
- For debugging purposes, you may find it helpful to
    plot the value of the rule-based output (-1, 0, 1) versus the stock price.

Strategy:

Use 2 indicators combined:
- RSI + Bollinger Bands

----------------------
Student Name: Xiao Nan (replace with your name)
GT User ID: nxiao30 (replace with your User ID)
GT ID: 903472104 (replace with your GT ID)
"""
import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

from util import get_data, plot_data, normalize, standard_score
from marketsimcode import compute_portvals, get_portfolio_stats, compute_daily_returns
from indicators import rsi, bollinger_bands

SYMBOL = "JPM"
IN_SAMPLE_DATES = (dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31))
OUT_SAMPLE_DATES = (dt.datetime(2010, 1, 1), dt.datetime(2011, 12, 31))
START_VAL = 100000

POSITIONS = [-1000, 0, 1000]


def testPolicy(symbol="JPM",
               sd=dt.datetime(2010, 1, 1),
               ed=dt.datetime(2011, 12, 31),
               sv=100000):
    """
    Params:
    @symbol: the stock symbol to act on
    @sd: A datetime object that represents the start date
    @ed: A datetime object that represents the end date
    @sv: Start value of the portfolio

    Return:
    @df_trades: A data frame whose values represent trades for each day.
        Legal values are +1000.0 indicating a BUY of 1000 shares,
        -1000.0 indicating a SELL of 1000 shares,
        and 0.0 indicating NOTHING.
        Values of +2000 and -2000 for trades are also legal
        so long as net holdings are constrained to -1000, 0, and 1000.
    """
    test_sd = sd
    test_ed = ed
    symbols = [symbol]

    # get price data
    df_prices = get_data(symbols, pd.date_range(test_sd, test_ed))
    # print(df_prices)
    # drop SPY
    df_prices.drop('SPY', axis=1, inplace=True)
    # normalize
    df_prices = normalize(df_prices)

    # Get bollinger_bands indicators bb_value, upper_band, lower_band
    (df_prices['bb_value'],
     df_prices['upper_band'],
     df_prices['lower_band']) = bollinger_bands(df_prices['JPM'])

    # Get bollinger_bands indicators bb_value, upper_band, lower_band
    df_prices['rsi'] = rsi(df_prices['JPM'])

    # fill na
    df_prices.fillna(0)

    # Find overbought region and change position to -1.0
    conditions = [
        ((df_prices['JPM'] >= df_prices['upper_band']) & (df_prices['rsi'] >= 70)),
        ((df_prices['JPM'] <= df_prices['lower_band']) & (df_prices['rsi'] <= 30))]
    choices = [-1.0, 1.0]
    df_prices['trade_target'] = np.select(conditions, choices, default=0.0)
    # print(df_prices)

    # get df_prices['holdings']
    df_prices['holdings'] = df_prices['trade_target']
    for i in range(1, len(df_prices)):
        if (df_prices.iloc[i, df_prices.columns.get_loc('holdings')] == 0) & (
                df_prices.iloc[i - 1, df_prices.columns.get_loc('holdings')] != 0):
            df_prices.iloc[i, df_prices.columns.get_loc('holdings')] = df_prices.iloc[
                i - 1, df_prices.columns.get_loc('holdings')]

    # get df_trades
    df_prices['trade'] = (df_prices['holdings'] - df_prices['holdings'].shift(1).fillna(0))
    df_prices['trade'] *= 1000
    # print(df_prices)

    df_trades = df_prices.drop(['JPM', 'bb_value', 'upper_band', 'lower_band', 'rsi',
                                'trade_target', 'holdings'], axis=1)

    # df_trades = df_prices.drop(df_prices[df_prices['position']==0.0].index)
    # print(df_trades)
    return df_trades


def convert_trades_to_order(df_trades):
    """ Convert df_trades to df_orders for calculating stats
    df_trades Example:
        Date,         trade
        2010-01-04,   +1000.0
    df_orders Example:
       Date,       Symbol, Order,  Shares
       2010-01-04, JPM,    BUY,    1000
    """
    df_orders = df_trades.copy()

    df_orders['Symbol'] = 'JPM'
    conditions = [
        (df_orders['trade'] > 0),
        (df_orders['trade'] < 0),
        (df_orders['trade'] == 0)]
    choices = ['BUY', 'SELL', 'HOLD']
    df_orders['Order'] = np.select(conditions, choices, default='HOLD')

    df_orders['Shares'] = np.abs(df_orders['trade'])

    # drop trade column
    df_orders.drop(['trade'], axis=1, inplace=True)

    # drop HOLD rows
    df_orders.drop(df_orders[df_orders['Order'] == "HOLD"].index, inplace=True)

    return df_orders


def benchmark(symbol, sd, ed, sv):
    """ Calculate benchmark strategy
    Benchmark strategy:
        Buy 1000 shares and hold, starting value 100000
    Order DataFrame:
        Date,       Symbol, Order,  Shares
        2010-01-04, JPM,    BUY,    1000

    Params:
    @symbol: the stock symbol to act on
    @sd: Start date of trading period
    @ed: End date of trading period
    @sv: Starting value

    Return:
    @df_benchmark: Dataframe for benchmark strategy
    """
    test_sd = sd
    test_ed = ed
    symbols = [symbol]

    # get price data
    df_prices = get_data(symbols, pd.date_range(test_sd, test_ed))
    # print(df_prices)
    start_date = min(df_prices.index)
    # print(start_date)

    orders = {'Date': [start_date],
              'Symbol': symbols,
              'Order': ['BUY'],
              'Shares': [1000]
              }

    df_orders = pd.DataFrame(orders)

    # df_orders['Date'] = pd.to_datetime(df_orders['Date'], format="%Y-%m-%d")
    df_orders.set_index('Date', inplace=True)
    # print("df_orders")
    # print(df_orders)
    # print(df_orders.info())

    # calcualte portfolio value based on trade orders (df_orders)
    portvals = compute_portvals(df_orders=df_orders, start_val=START_VAL, commission=9.95, impact=0.005,
                                start_date=sd, end_date=ed)

    # normalize the portvals
    portvals = normalize(portvals)

    cr, adr, sddr, sr = get_portfolio_stats(portvals)

    return portvals, cr, adr, sddr, sr


def print_compare_benchmark(start_date, end_date, benchmark_sharpe_ratio, sharpe_ratio_manual_strategy,
                            benchmark_cum_ret, cum_ret_manual_strategy, benchmark_std_daily_ret,
                            std_daily_ret_manual_strategy, benchmark_avg_daily_ret, avg_daily_ret_manual_strategy,
                            benchmark_portvals, manual_strategy_portvals):
    # Compare portfolio against $SPX
    print("Date Range: {} to {}".format(start_date, end_date))
    print()
    print("Sharpe Ratio of benchmark: {}".format(benchmark_sharpe_ratio))
    print("Sharpe Ratio of manual_strategy : {}".format(sharpe_ratio_manual_strategy))
    print()
    print("Cumulative Return of benchmark: {}".format(benchmark_cum_ret))
    print("Cumulative Return of manual_strategy : {}".format(cum_ret_manual_strategy))
    print()
    print("Standard Deviation of benchmark: {}".format(benchmark_std_daily_ret))
    print("Standard Deviation of manual_strategy : {}".format(std_daily_ret_manual_strategy))
    print()
    print("Average Daily Return of benchmark: {}".format(benchmark_avg_daily_ret))
    print("Average Daily Return of manual_strategy : {}".format(avg_daily_ret_manual_strategy))
    print()
    print("Final benchmark Portfolio Value: {}".format(benchmark_portvals[-1]))
    print("Final manual_strategy Portfolio Value: {}".format(manual_strategy_portvals[-1]))


def test_code():
    start_date = OUT_SAMPLE_DATES[0]
    end_date = OUT_SAMPLE_DATES[1]

    df_trades = testPolicy(symbol=SYMBOL, sd=start_date, ed=end_date, sv=START_VAL)
    # print("df_trades")
    # print(df_trades.head())
    # print(df_trades.tail())

    df_orders = convert_trades_to_order(df_trades)
    # print("df_orders")
    # print(df_orders)

    short_entry_points = df_orders.index[df_orders['Order'] == 'SELL'].tolist()
    # print("short_entry_points", short_entry_points)

    long_entry_points = df_orders.index[df_orders['Order'] == 'BUY'].tolist()
    # print("long_entry_points", long_entry_points)

    # calcualte portfolio value based on trade orders (df_orders)
    manual_strategy_portvals = compute_portvals(df_orders=df_orders, start_val=START_VAL, commission=9.95, impact=0.005,
                                                start_date=start_date, end_date=end_date)

    # normalize the portvals
    manual_strategy_portvals = normalize(manual_strategy_portvals)

    (cum_ret_manual_strategy, avg_daily_ret_manual_strategy,
     std_daily_ret_manual_strategy, sharpe_ratio_manual_strategy) = get_portfolio_stats(manual_strategy_portvals)

    # Get stats for benchmark performance
    (benchmark_portvals, benchmark_cum_ret, benchmark_avg_daily_ret,
     benchmark_std_daily_ret, benchmark_sharpe_ratio) = benchmark(symbol=SYMBOL,
                                                                  sd=start_date,
                                                                  ed=end_date,
                                                                  sv=START_VAL)

    print_compare_benchmark(start_date, end_date, benchmark_sharpe_ratio, sharpe_ratio_manual_strategy,
                            benchmark_cum_ret, cum_ret_manual_strategy, benchmark_std_daily_ret,
                            std_daily_ret_manual_strategy, benchmark_avg_daily_ret, avg_daily_ret_manual_strategy,
                            benchmark_portvals, manual_strategy_portvals)

    plot_compare(benchmark_portvals, manual_strategy_portvals, long_entry_points, short_entry_points)


def plot_compare(benchmark_portvals, manual_strategy_portvals,
                 long_entry_points, short_entry_points):
    """Plot portfolio values of Benchmark and Manual Strategy"""
    final_df = pd.concat([benchmark_portvals, manual_strategy_portvals], axis=1)
    final_df.columns = ['Normalized Benchmark Portfolio Value',
                        'Normalized Manual Strategy Portfolio Value']
    # print(final_df)
    # Plot final dataframe
    title = "Benchmark vs Manual Strategy"
    xlabel = "Date"
    ylabel = "Portfolio Value"
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = final_df.plot(title=title, fontsize=12, color=['g', 'r'], figsize=(10, 6))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.axhline(y=1., color='m', linestyle=':')

    # add blue lines for LONG entry points
    for x in long_entry_points:
        plt.axvline(x=x, color='b', linestyle='--')

    # add black lines for SHORT entry points
    for x in short_entry_points:
        plt.axvline(x=x, color='k', linestyle='--')

    plt.show()
    return None


if __name__ == "__main__":
    test_code()