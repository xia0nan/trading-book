import pandas as pd
import datetime as dt
from util import get_data, plot_data


def compute_portvals(orders_file="../orders/orders.csv", start_val=1000000, commission=9.95, impact=0.005):
    # NOTE: orders_file may be a string, or it may be a file object.

    # 1. Import Order dataframe
    df_orders = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan'])
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

    # 4. Get date range
    start_date = min(df_orders.index)
    end_date = max(df_orders.index)
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
        transaction_volume = sign * int(row['Shares'])
        df_trades.loc[[index], [row['Symbol']]] += transaction_volume
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
    of = "../orders/orders-02.csv"
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


if __name__ == "__main__":
    test_code()
# Ref. YouTube: https://youtu.be/1ysZptg2Ypk
# Ref. YouTube: https://youtu.be/TstVUVbu-Tk