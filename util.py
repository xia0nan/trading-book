import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

CURRENT_DIR = Path.cwd()


def normalize(prices):
    """Normalize pandas DataFrame by divide by first row"""
    return prices / prices.iloc[0]


def standard_score(df):
    """ Calculate z-score of columns
    z = (x - mu) / sd
    mu is the mean of the population.
    sd is the standard deviation of the population.

    Use axis=0 get column mean and std

    Params:
    @df: input score, Pandas DataFrame or Series

    Return:
    @z: output standardlized score
    """
    return (df - df.mean(axis=0)) / df.std(axis=0)

def symbol_to_path(symbol, base_dir=None):
    """Return CSV file path given ticker symbol."""
    if base_dir is None:
        base_dir = os.environ.get("MARKET_DATA_DIR", '../data/')
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def get_data(symbols, dates, addSPY=True, colname='Adj Close'):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)

    # add SPY for reference, if absent
    if addSPY and 'SPY' not in symbols:
        # handles the case where symbols is np array of 'object'
        symbols = ['SPY'] + list(symbols)

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol),
                              index_col='Date',
                              parse_dates=True,
                              usecols=['Date', colname],
                              na_values=['nan'])
        df_temp = df_temp.rename(columns={colname: symbol})
        df = df.join(df_temp)
        # drop dates SPY did not trade
        if symbol == 'SPY':
            df = df.dropna(subset=["SPY"])

    return df


def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()


def get_orders_data_file(basefilename):
    return open(os.path.join(os.environ.get("ORDERS_DATA_DIR", 'orders/'), basefilename))


def get_learner_data_file(basefilename):
    return open(os.path.join(os.environ.get("LEARNER_DATA_DIR", 'Data/'), basefilename), 'r')
