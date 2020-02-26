import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")

import datetime as dt
import pandas as pd
import numpy as np

import random

import matplotlib.pyplot as plt

import util as ut
import QLearner as ql
import RTLearner as rtl
import BagLearner as bl
import ManualStrategy as ms
import indicators as ind
import marketsimcode as sim

pd.options.mode.chained_assignment = None

SYMBOL = "JPM"
IN_SAMPLE_DATES = (dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31))
OUT_SAMPLE_DATES = (dt.datetime(2010, 1, 1), dt.datetime(2011, 12, 31))
START_VAL = 100000
POSITIONS = [-1000, 0, 1000]
ACTIONS = [-1, 0, 1]  # [SHORT, CASH, LONG]

COMMISSIONS = 0.0
IMPACT = 0.0

X_FEATURES = ['upper_band', 'lower_band', 'RSI']
Y_WINDOW = 9

LEAF_SIZE = 5
NUM_OF_BAGS = 20

YBUY = 0.04  # long if 5% return in 20 days
YSELL = -0.04  # short if 5% loss in 20 days

SEED = 42


class StrategyLearner(object):

    # constructor
    def __init__(self, verbose=False, impact=IMPACT):
        self.verbose = verbose
        self.impact = impact
        # self.learner = ql.QLearner(num_states=1000, num_actions=3)
        self.learner = bl.BagLearner(learner=rtl.RTLearner,
                                     kwargs={"leaf_size": LEAF_SIZE},
                                     bags=NUM_OF_BAGS,
                                     boost=False,
                                     verbose=False)

    # this method should create a QLearner, and train it for trading
    def addEvidence(self,
                    symbol=SYMBOL,
                    sd=IN_SAMPLE_DATES[0],
                    ed=IN_SAMPLE_DATES[1],
                    sv=START_VAL):
        """
        Your learner will be provided with a stock symbol and a time period.
        It should use this data to learn a strategy.
        For instance, for a regression-based learner it will use this data
            to make predictions about future price changes.

        Params
        @verbose: if False do not generate any output
        @impact: The market impact of each transaction.
        @symbol: the stock symbol to train on
        @sd: A datetime object that represents the start date
        @ed: A datetime object that represents the end date
        @sv: Start value of the portfolio
        """

        prices = self.get_prices(symbol, sd, ed)

        # Get df_indicators
        df_indicators = ind.get_df_indicators(prices, plot_df=False, dropNA=False)
        # print(df_indicators)

        # Get trainX and trainY
        trainX, trainY = self.get_train_X_Y(df_indicators)
        # print(df_indicators)

        # add training data
        self.learner.addEvidence(trainX, trainY)

        # check optimal performance
        # print("training optimal performance")
        # self.check_performance(df_indicators, sd, ed)

    def get_prices(self, symbol, sd, ed):
        """ Get price data based on symbol and start_date, end_date """

        # get symbol list and date range
        syms = [symbol]
        dates = pd.date_range(sd, ed)

        # Get prices data, automatically adds SPY
        prices_all = ut.get_data(syms, dates)

        # normalize price, price[t] /= price[0]
        prices_all = ind.normalize(prices_all)

        prices = prices_all[syms]  # only portfolio symbols
        prices_SPY = prices_all['SPY']  # only SPY, for comparison later
        # if self.verbose: print prices

        return prices

    def get_train_X_Y(self, df_indicators, window=Y_WINDOW):
        """ Split df_indicators into trainX and trainY

        trainX: ['sma', 'bb_value', 'RSI', 'momentum', 'volatility']

        Your code should classify based on N day change in price.
        You need to build a new Y that reflects the N day change and aligns with the current date.
        Here's pseudo code for the calculation of Y
            ret = (price[t+N]/price[t]) - 1.0
            if ret > YBUY:
                Y[t] = +1 # LONG
            else if ret < YSELL:
                Y[t] = -1 # SHORT
            else:
                Y[t] = 0 # CASH

        """

        # Get trainX from dataframe to ndarray
        trainX_df = df_indicators[X_FEATURES]
        # remove window in the end
        trainX_df.drop(trainX_df.tail(Y_WINDOW).index, inplace=True)
        trainX_df.dropna(inplace=True)
        # print(trainX_df)
        trainX = trainX_df.values
        # print(trainX)

        df_indicators['price_tn'] = df_indicators['prices'].shift(-Y_WINDOW)
        df_indicators['ret'] = df_indicators['price_tn'] / df_indicators['prices'] - 1.0

        # get Y based on N day return
        conditions = [
            (df_indicators['ret'] > YBUY + self.impact),
            (df_indicators['ret'] < YSELL - self.impact)]
        choices = [1, -1]
        choices_name = ["LONG", "SHORT"]
        df_indicators['position'] = np.select(conditions, choices, default=0)
        df_indicators['position_name'] = np.select(conditions, choices_name, default="CASH")

        df_indicators.dropna(inplace=True)
        # print(df_indicators.dtypes)

        # get Y as ndarray of target positions
        trainY = df_indicators['position'].values
        # print(trainY)

        return trainX, trainY

    def get_orders_from_position(self, df_indicators, col_name='position'):
        """ Get orders dataframe to evaluate performance based on holdings"""
        df_indicators['trade'] = (df_indicators[col_name] - df_indicators[col_name].shift(1).fillna(0))
        df_indicators['trade'] *= 1000
        df_trades = df_indicators[['trade']]
        df_orders = self.convert_trades_to_order(df_trades)
        return df_orders

    def convert_trades_to_order(self, df_trades):
        """ Convert df_trades to df_orders for calculating stats
        df_trades Example:
            Date,         trade
            2010-01-04,   +1000.0
        df_orders Example:
        Date,       Symbol, Order,  Shares
        2010-01-04, JPM,    BUY,    1000
        """
        df_orders = df_trades.copy()

        df_orders['Symbol'] = SYMBOL
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

    @staticmethod
    def normalize_df(prices):
        """Normalize pandas DataFrame by divide by first row"""
        return prices / prices.iloc[0]

    def check_performance(self, df_indicators, sd, ed, col_name='position'):
        df_orders = self.get_orders_from_position(df_indicators, col_name=col_name)
        # print("df_orders")
        # print(df_orders)

        strategy_portvals = sim.compute_portvals(df_orders=df_orders,
                                                 start_val=START_VAL,
                                                 commission=0.0,
                                                 impact=self.impact,
                                                 start_date=sd,
                                                 end_date=ed)

        # normalize the portvals
        strategy_portvals = self.normalize_df(strategy_portvals)

        (cum_ret, avg_daily_ret, std_daily_ret,
         sharpe_ratio) = sim.get_portfolio_stats(strategy_portvals)

        print("final strategy_portvals", strategy_portvals[-1])
        print("cum_ret", cum_ret)
        print("avg_daily_ret", avg_daily_ret)
        print("std_daily_ret", std_daily_ret)
        print("sharpe_ratio", sharpe_ratio)

    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol=SYMBOL,
                   sd=OUT_SAMPLE_DATES[0],
                   ed=OUT_SAMPLE_DATES[1],
                   sv=START_VAL):
        """
        Your learner will be provided a symbol and a date range.
        All learning should be turned OFF during this phase.

        The requirement that consecutive calls to testPolicy() produce
            the same output for the same input means that
            you cannot update, train, or tune your learner in this method.
        For example, a solution that uses Q-Learning should use
            querySetState() and not query() in testPolicy().
        Updating, training, and tuning (query()) is fine inside addEvidence().

        Params
        @verbose: if False do not generate any output
        @impact: The market impact of each transaction.
        @symbol: the stock symbol to train on
        @sd: A datetime object that represents the start date
        @ed: A datetime object that represents the end date
        @sv: Start value of the portfolio

        Return
        @df_trades: A data frame whose values represent trades for each day.
            Legal values are +1000.0 indicating a BUY of 1000 shares,
            -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.
            Values of +2000 and -2000 for trades are also legal when switching
            from long to short or short to long so long as net holdings are
            constrained to -1000, 0, and 1000.
        """
        # here we build a fake set of trades
        # your code should return the same sort of data
        prices = self.get_prices(symbol, sd, ed)
        # print("prices.shape", prices.shape)

        # Get df_indicators
        df_indicators = ind.get_df_indicators(prices, plot_df=False, dropNA=True)
        # print(df_indicators.shape)

        # Get testX from dataframe to ndarray
        testX_df = df_indicators[X_FEATURES]
        # remove window in the end
        # testX_df.drop(testX_df.tail(Y_WINDOW).index, inplace=True)
        # testX_df.dropna(inplace=True)
        # print("testX_df.shape", testX_df.shape)
        testX = testX_df.values
        # print(testX)

        # get testY
        df_indicators['position'] = self.learner.query(df_indicators[X_FEATURES].values)

        # testY=self.learner.query(testX)
        # print(testY)
        # print(testY.shape)
        df_indicators['position'] = np.round(df_indicators['position'])
        # print type(testY)

        # df_indicators.dropna(inplace=True)
        # print("df_indicators.shape", df_indicators.shape)

        df_indicators['position'] = df_indicators['position'].astype(np.int64)

        # assign to testY
        testY = df_indicators['position']

        # print(df_indicators)
        # print("test performance")
        # self.check_performance(df_indicators, sd, ed)

        df_indicators['trade'] = (df_indicators['position'] - df_indicators['position'].shift(1).fillna(0))
        df_indicators['trade'] *= 1000
        df_trades = df_indicators[['trade']]

        # if self.verbose: print type(trades)  # it better be a DataFrame!
        # if self.verbose: print trades.shape
        # if self.verbose: print trades
        # if self.verbose: print prices_all
        return df_trades


def test_code():
    np.random.seed(SEED)
    random.seed(SEED)

    learner = StrategyLearner(verbose=False, impact=IMPACT)
    learner.addEvidence()

    # test in sample
    learner.testPolicy(sd=IN_SAMPLE_DATES[0], ed=IN_SAMPLE_DATES[1])

    # # test outsample
    # learner.testPolicy()


if __name__ == "__main__":
    test_code()