import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

from util import get_data, plot_data

SYMBOL = "JPM"
IN_SAMPLE_DATES = (dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31))
OUT_SAMPLE_DATES = (dt.datetime(2010, 1, 1), dt.datetime(2011, 12, 31))
START_VAL = 100000
POSITIONS = [-1000, 0, 1000]  # 1000 shares short, 0 shares, 1000 shares long


def sma(prices, n=20):
    """ Calculate simple moving average of one stock using rolling mean
    Params:
    @prices: price data of stock with dates as index
    @n: number of days in smoothing period (typically 20)

    Return:
    @moving_avg: Moving average price of the stock
    """
    moving_avg = prices.rolling(window=n).mean()
    return moving_avg


def bollinger_bands(prices, n=20, m=2):
    """ Calculate Bollinger Band using moving average and SD
    Reference: https://www.investopedia.com/terms/b/bollingerbands.asp

    Upper_band = moving_average + m * std(price)
    Lower_band = moving_average - m * std(price)

    BB Value:
        %B = (Current Price - Lower Band) / (Upper Band - Lower Band)

    Params:
    @prices: price data of stock with dates as index
    @n: number of days in smoothing period (typically 20)
    @m: number of standard deviations (typicallly 2)

    Return:
    @bb_value: BB value of Bollinger bands
    @upper_band: Upper Bollinger Band
    @lower_band: Lower Bollinger Band
    """

    # rolling mean
    ma = sma(prices, n)
    # rolling std
    std = prices.rolling(window=n).std()
    # upper band & lower band
    upper_band = ma + m * std
    lower_band = ma - m * std
    # bb value
    bb_value = (prices - lower_band) / (upper_band - lower_band)

    # bb value 2
    # bb_value[t] = (price[t] - SMA[t])/(2 * stdev[t])
    # bb_value = (prices - ma) / (2 * std)

    return bb_value, upper_band, lower_band


def rsi(prices, period=14):
    """ Calculate Relative Strength Index - RSI
    Momentum indicator that measures the magnitude of recent price changes
        to evaluate overbought or oversold
    The RSI compares bullish and bearish price momentum
        plotted against the graph of an asset's price.
    Signals are considered overbought when the indicator is above 70%
        and oversold when the indicator is below 30%.

    RSI = 100 - 100 / (1 + RS)
    RS = avg_gain / avg_loss

    The very first calculations for average gain and average loss
        are simple 14-period averages:
    First Average Gain = Sum of Gains over the past 14 period / 14
    First Average Loss = Sum of Losses over the past 14 period / 14

    The second, and subsequent, calculations are based on
        the prior averages and the current gain loss:
    Average Gain = [(previous Average Gain) x 13 + current Gain] / 14
    Average Loss = [(previous Average Loss) x 13 + current Loss] / 14

    Params:
    @prices: price data of stock with dates as index
    @period: number of days in a period

    Returns:
    @rsi: Relative Strength Index of stock
    """

    df_rsi = pd.DataFrame(index=prices.index)
    df_rsi['price'] = prices
    # calculate price change
    df_rsi['chg'] = df_rsi['price'].diff()
    # get gain or loss based on price change
    df_rsi['gain'] = np.where(df_rsi['chg'] > 0, df_rsi['chg'], 0)
    df_rsi['loss'] = np.where(df_rsi['chg'] < 0, df_rsi['chg'], 0)
    # use 14days rolling mean
    df_rsi['avg_gain'] = df_rsi['gain'].rolling(window=period).mean()
    df_rsi['avg_loss'] = np.abs(df_rsi['loss'].rolling(window=period).mean())
    # calculate RS
    df_rsi['RS'] = df_rsi['avg_gain'] / df_rsi['avg_loss']
    # calculate RSI
    df_rsi['RSI'] = 100 - 100 / (1 + df_rsi['RS'])

    # print(df_rsi)
    return df_rsi['RSI']


def momentum(prices, n=10):
    """ Calculate 10 days momentum of the price

    Momentum = V - Vx
    where:
    V  = Latest price
    Vx = Closing price
    x  = Number of days ago

    Formula:
    momentum[t] = (price[t]/price[t-N]) - 1

    Params:
    @prices: Price data of stock with dates as index
    @n: number of days to calculate momumtum

    Return:
    @momentum: Momentum of the price

    """
    df_momentum = pd.DataFrame(index=prices.index)
    # get price[t-N]
    df_momentum['pricet'] = prices
    # get price[t]
    df_momentum['pricet_n'] = prices.shift(n)
    # get price difference comparing to n days ago
    df_momentum['price_diff'] = prices.diff(n)
    # Calculate momentum
    df_momentum['momentum'] = (df_momentum['pricet'] / df_momentum['pricet_n']) - 1.
    # Standardized momentum using z-score
    df_momentum['momentum_z'] = standard_score(df_momentum['momentum'])

    # print(df_momentum)
    return df_momentum['momentum_z']


def volatility(prices, period=20):
    """ Volatility is just the stdev of daily returns. """
    df_var = pd.DataFrame(index=prices.index)
    df_var['price'] = prices

    df_var['daily_return'] = compute_daily_returns(prices)
    df_var['volatility'] = df_var['daily_return'].rolling(window=period).std()

    # Standardized volatility using z-score
    df_var['volatility_z'] = standard_score(df_var['volatility'])

    # print(df_var)
    return df_var['volatility_z']


def test_code():
    """
    Technical indicators:
    Ref. https://medium.com/@harrynicholls/7-popular-technical-indicators-and-how-to-use-them-to-increase-your-trading-profits-7f13ffeb8d05
        Trend:
            - SMA
                - If you were using price/SMA as an indicator
                    you would want to create a chart with 3 lines: Price, SMA, Price/SMA.
                    In order to facilitate visualization of the indicator you might normalize the data
                    to 1.0 at the start of the date range  (i.e. divide price[t] by price[0]).
            - Moving Average Convergence Divergence (MACD)
            - Parabolic Stop and Reverse (SAR)
        Volatility:
            - Bollinger Bands
                - you might create a chart that shows the price history of the stock,
                    along with "helper data" (such as upper and lower bollinger bands)
                    and the value of the indicator itself.
            - AVERAGE TRUE RANGE (ATR)
            - DONCHIAN CHANNELS (DC)
            - KELTNER CHANNELS (KC)
        Momentum:
            - RSI(Relative Strength Index)
            - Ichimoku Kinko Hyo (AKA Ichimoku Cloud)
            - Stochastic
            - Average Directional Index (ADX)
        Volume:
            - On-Balance Volume
            - Chaikin Money Flow
            - Klinger Volume Oscillator


    Return:
        @df_indicators: technical indicators dataframe for a stock
    """

    # 1. Read data using get_data()
    train_sd = IN_SAMPLE_DATES[0]
    train_ed = IN_SAMPLE_DATES[1]
    symbols = [SYMBOL]
    # get price data
    df_prices = get_data(symbols, pd.date_range(train_sd, train_ed))
    # print(len(df_prices))

    # Check if missing data
    # Check JPM null
    spy = get_data(['SPY'], pd.date_range(train_sd, train_ed))
    assert len(spy) == len(df_prices)
    # print(df_prices.isnull().values.any())

    # Get pd.Series SPY
    price_SPY = df_prices['SPY']
    # print(price_SPY.head())
    # print(type(price_SPY))

    # Get pd.Series JPM
    price_JPM = df_prices['JPM']
    # print(price_JPM.head())
    # print(type(price_JPM))

    # Normalize data
    price_SPY = normalize(price_SPY)
    price_JPM = normalize(price_JPM)
    # print(price_SPY_norm.head())
    # print(price_JPM_norm.head())

    # 2. initialize df_indicators, every column to save indicator value
    df_indicators = pd.DataFrame(index=df_prices.index)
    df_indicators['prices'] = price_JPM

    # 3. Get moving average
    rolling_window = 20

    mv_avg = sma(price_JPM, rolling_window)
    # print(mv_avg)

    df_indicators['sma'] = mv_avg
    # print(df_indicators)

    # 4. Get Bollinger Band
    bb_value, upper_band, lower_band = bollinger_bands(price_JPM, n=rolling_window)
    # update df_indicators
    df_indicators['bb_value'] = bb_value
    df_indicators['upper_band'] = upper_band
    df_indicators['lower_band'] = lower_band
    # print(df_indicators)

    # 5. Get RSI(Relative Strength Index)
    rsi_value = rsi(price_JPM)
    # update df_indicators
    df_indicators['RSI'] = rsi_value
    # print(df_indicators)

    # 6. Get momentum
    momentum_value = momentum(price_JPM)
    df_indicators['momentum'] = momentum_value
    # print(df_indicators)

    # 7. Get volatility
    df_indicators['volatility'] = volatility(price_JPM)
    # print(df_indicators)

    # 8. Plot indicators
    plot_indicators(df_indicators)


def plot_indicators(df_indicators):
    """ Plot indicators in df_indicators"""
    # 8.1 Plot Price chart for comparison later
    df_price = df_indicators[['prices']].copy()
    df_price.rename(columns={'prices': 'JPM stock price'}, inplace=True)
    plot_data(df_price, title="JPM stock price normalized", xlabel="Date", ylabel="Price", starting_ref=1.0)

    # 8.1 Plot Price, SMA, Price/SMA
    df_sma = df_indicators[['prices', 'sma']].copy()
    df_sma['Price to SMA ratio'] = df_sma['prices'] / df_sma['sma']
    df_sma.rename(columns={'prices': 'JPM stock price', 'sma': '20-days simple moving average'},
                  inplace=True)
    plot_data(df_sma, title="Price / SMA Ratio", xlabel="Date", ylabel="Price/SMA", starting_ref=1.0)

    # 8.2 Plot Bollinger Band
    df_bb = df_indicators[['prices', 'sma', 'lower_band', 'upper_band']].copy()
    df_bb.rename(columns={'lower_band': 'Lower Bollinger Band',
                          'upper_band': 'Upper Bollinger Band',
                          'sma': '20-days simple moving average'},
                 inplace=True)
    plot_data(df_bb, title="Bollinger Band", xlabel="Date", ylabel="Bollinger Band", starting_ref=1.0)

    df_bb_value = df_indicators[['bb_value']].copy()
    df_bb_value.rename(columns={'bb_value': 'Bollinger Band BB Value'},
                       inplace=True)
    plot_data_with_line(df_bb_value, title="Bollinger Band BB Value", xlabel="Date", ylabel="BB Value",
                        upper_y=1.0, lower_y=0.0, middle_y=0.5)

    # 8.3 Plot RSI
    df_rsi = df_indicators[['RSI']].copy()
    df_rsi.rename(columns={'RSI': 'Relative Strength Index (RSI)'},
                  inplace=True)
    plot_data_with_line(df_rsi, title="Relative Strength Index (RSI)", xlabel="Date", ylabel="RSI",
                        upper_y=70, lower_y=30, middle_y=50)

    # 8.4 Plot Momentum
    df_momentum = df_indicators[['momentum']].copy()
    df_momentum.rename(columns={'momentum': '14 days Momentum'},
                       inplace=True)
    plot_data_with_line(df_momentum, title="14 days Momentum", xlabel="Date", ylabel="Momentum",
                        upper_y=1.0, lower_y=-1.0, middle_y=0.)

    # 8.5 Plot Volatility
    df_volatility = df_indicators[['volatility']].copy()
    df_volatility.rename(columns={'volatility': '20 days Volatility'},
                         inplace=True)
    plot_data_with_line(df_volatility, title="20 days Volatility", xlabel="Date", ylabel="Volatility",
                        upper_y=1.0, lower_y=-1.0, middle_y=0.)


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


# From previous exercise
def compute_daily_returns(df):
    """Compute and return the daily return values."""
    # Note: Returned DataFrame must have the same number of rows
    return ((df / df.shift(1)) - 1).fillna(0)


def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price", starting_ref=1.0):
    """Plot stock prices with a custom title and meaningful axis labels."""
    plt.close()
    ax = df.plot(title=title, fontsize=12, figsize=(10, 6))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.axhline(y=starting_ref, color='y', linestyle=':')
    plt.show()


def plot_data_with_line(df, title="Stock prices", xlabel="Date", ylabel="Price",
                        upper_y=1.0, lower_y=0.0, middle_y=None):
    """Plot stock prices with a custom title and meaningful axis labels."""
    plt.close()
    ax = df.plot(title=title, fontsize=12, figsize=(10, 6))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.axhline(y=upper_y, color='r', linestyle='--')
    plt.axhline(y=lower_y, color='g', linestyle='--')
    plt.axhline(y=middle_y, color='y', linestyle=':')

    plt.show()


if __name__ == "__main__":
    test_code()
