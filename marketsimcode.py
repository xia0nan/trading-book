"""
An improved version of your marketsim code that accepts
    a "trades" data frame (instead of a file).
More info on the trades data frame below.
"""
import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")

import pandas as pd
import numpy as np
import datetime as dt
from util import get_data, plot_data


def compute_portvals(df_orders=None, orders_file=None,
                     start_val=1000000, commission=9.95, impact=0.005,
                     start_date=None, end_date=None):
    # NOTE: orders_file may be a string, or it may be a file object.

    # 1. Import Order dataframe
    # df_orders = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan'])
    # print("df_orders")
    # print(df_orders.head())

    # 2. Sort order file by dates (ascending)
    df_orders = df_orders.sort_index(ascending=1)
    # print("df_orders")
    # print(df_orders)
    # print(df_orders.shape)

    # 3. Get symbols for the portfolio
    symbols = df_orders["Symbol"].unique().tolist()
    # print("symbols")
    # print(symbols)
    # print(type(symbols))

    # 4. Get date range.
    # start_date = min(df_orders.index)
    # end_date = max(df_orders.index)
    # print("start_date", start_date)
    # print(type(start_date))
    # print("end_date", end_date)
    # print(type(end_date))

    # 5. Get df_prices using adjusted closing price, add cash column at last (all 1.0)
    df_prices = get_data(symbols, pd.date_range(start_date, end_date))
    # sort by date
    df_prices = df_prices.sort_index(ascending=1)
    # print("df_prices")
    # print(df_prices.head())
    df_prices_SPY = df_prices['SPY']
    # drop SPY column
    df_prices = df_prices.drop(['SPY'], axis=1)
    # add cash column
    df_prices['CASH'] = 1.0
    # add index name
    df_prices.index.name = 'DATE'
    # print(df_prices)

    # 6. Get df_trades using df_orders and df_prices
    df_trades = pd.DataFrame(0., columns=df_prices.columns, index=df_prices.index)
    # print("df_trades")
    # print(df_trades)
    for index, row in df_orders.iterrows():
        # print(df_prices.loc[[index], [row['Symbol']]])

        # if SELL symbol volume should be x(-1), otherwise x1
        sign = -1.0 if row['Order'] == 'SELL' else 1.0
        # print("sign", sign)
        transaction_volume = sign * int(row['Shares'])
        # print("transaction_volume", transaction_volume)
        df_trades.loc[[index], [row['Symbol']]] += transaction_volume
        # print("df_trades", df_trades)
        # CASH is changing in the opposite direction, x (-1), could be multiple trades per day so use "+=" to update
        transaction_price = df_prices.loc[[index], [row['Symbol']]].values[0]
        df_trades.loc[[index], ['CASH']] += transaction_price * (-1) * transaction_volume

        ## Part 2 Transaction Costs
        # 2.1 Deduct transaction commission from CASH account for each trade
        df_trades.loc[[index], ['CASH']] -= commission

        # 2.2 Deduct market impact from CASH account for each trade
        market_impact = int(row['Shares']) * transaction_price * impact
        df_trades.loc[[index], ['CASH']] -= market_impact

    # print("df_trades")
    # print(df_trades)

    # 7. Get df_holdings
    df_holdings = pd.DataFrame(0., columns=df_trades.columns, index=df_trades.index)
    # print("df_holdings")
    # print(df_holdings)

    # initialize first row of df_holdings
    df_holdings.iloc[0] = df_trades.iloc[0]
    df_holdings.iloc[0]['CASH'] += start_val
    # print(df_holdings.iloc[0])

    for i in range(1, len(df_holdings)):
        df_holdings.iloc[i] = df_holdings.iloc[i - 1] + df_trades.iloc[i]
    # print("df_holdings")
    # print(df_holdings)

    # 8. Get df_values SUM(symbol_volume * symbol price) using df_holdings & df_prices
    df_values = pd.DataFrame(0., columns=df_holdings.columns, index=df_holdings.index)
    # print("df_values")
    # print(df_values)

    # Use element-wise multiplication
    df_values = df_prices * df_holdings
    # print(df_values)

    # 9. Get portvals by using row sum of df_values (axis=1)
    portvals = df_values.sum(axis=1)
    # print("portvals")
    # print(portvals)

    return portvals


def test_code():
    # Define input parameters
    of = "./orders/orders-02.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file=of, start_val=sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2008, 6, 1)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [
        0.2, 0.01, 0.02, 1.5]
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [
        0.2, 0.01, 0.02, 1.5]

    # Compare portfolio against $SPX
    print("Date Range: {} to {}".format(start_date, end_date))
    print()
    print("Sharpe Ratio of Fund: {}".format(sharpe_ratio))
    print("Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY))
    print()
    print("Cumulative Return of Fund: {}".format(cum_ret))
    print("Cumulative Return of SPY : {}".format(cum_ret_SPY))
    print()
    print("Standard Deviation of Fund: {}".format(std_daily_ret))
    print("Standard Deviation of SPY : {}".format(std_daily_ret_SPY))
    print()
    print("Average Daily Return of Fund: {}".format(avg_daily_ret))
    print("Average Daily Return of SPY : {}".format(avg_daily_ret_SPY))
    print()
    print("Final Portfolio Value: {}".format(portvals[-1]))


def get_portfolio_stats(port_val, daily_rf=0., samples_per_year=252):
    """ Calculate statistics on daily portfolio value, given daily risk-free rate and data sampling frequency.
        Params:
        @port_val: Daily Portfolio value
        @daily_rf: Daily risk free rate
        @samples_per_year: Samples per year

        Return: a tuple consisting of the following statistics (in order):
        @cr: cumulative return,
        @adr: average daily return,
        @sddr: standard deviation of daily return,
        @sr: Sharpe ratio
        Note: The return statement provided ensures this order.
    """

    # Get portfolio statistics (note: std_daily_ret = volatility)
    daily_return = compute_daily_returns(port_val)[1:]
    cr = get_cumulative_return(port_val)
    adr = daily_return.mean()
    sddr = daily_return.std()

    # Sharpe ratio = K * (Return of Portfolio - Risk-free Rate) / SD of portfolio's excess return
    # Daily sampling K = sqrt(252.)
    sr = np.sqrt(samples_per_year) * (adr - daily_rf) / sddr

    return cr, adr, sddr, sr


def get_portfolio_value(prices, allocs, start_val):
    """ Compute daily portfolio value given stock prices, allocations and starting value.

        Return: pandas Series or DataFrame (with a single column)
    """
    return (prices * allocs).sum(1) * start_val


def get_cumulative_return(df):
    """ Get Cumulative return of Portfolio"""
    return df[-1] / df[0] - 1.


def compute_daily_returns(df):
    """Compute and return the daily return values."""
    # Note: Returned DataFrame must have the same number of rows
    return ((df / df.shift(1)) - 1).fillna(0)


def normalize_df(port_val):
    """ Return normalized dataframe starting with 1.0 """
    return port_val / port_val.ix[0, :]


def fill_price_df(prices):
    """ Fill forward and then fill backward price data """
    prices.fillna(method='ffill', inplace=True)
    prices.fillna(method='bfill', inplace=True)
    return prices


if __name__ == "__main__":
    test_code()
# Ref. YouTube: https://youtu.be/1ysZptg2Ypk
# Ref. YouTube: https://youtu.be/TstVUVbu-Tk