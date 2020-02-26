# Trading Book

Jupyter notebook for algo trading exploration

## Getting Started

These instructions will get you a conda environment of the project up and running.

### Prerequisites
* System: Ubuntu 18.04 LTS
* Anaconda3
* Python 3.7

```
- Go to link: https://www.anaconda.com/distribution/ 
- Install python3 version of latest Anaconda3
```

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
sudo apt update
sudo apt upgrade
sudo apt autoremove

cd /tmp
curl -O https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
sha256sum Anaconda3-2019.10-Linux-x86_64.sh
bash Anaconda3-2019.10-Linux-x86_64.sh
source ~/.bashrc

conda create --name trading_book python=3.7
conda activate trading_book
pip install -U pip
pip install -r requirements.txt
```

## Pipeline
### Example
1. Raw Price Data

|Date      |Open |High |Low  |Close|Volume  |Adj Close|
|----------|-----|-----|-----|-----|--------|---------|
|2012-09-12|39.88|40.25|39.79|39.92|29882200|39.92    |
|2012-09-11|38.80|39.63|38.73|39.60|22378900|39.60    |
|2012-09-10|39.23|39.78|38.73|38.76|20853500|38.76    |

2. Get Price DataFrame

|Date      |Symbol|Close|
|----------|------|-----|
|2012-09-12|SPY   |39.92|
|2012-09-11|SPY   |39.60|
|2012-09-10|SPY   |38.76|

3. Get Technical Indicators

|Date      |Symbol|Close|RSI|MA_20|
|----------|------|-----|---|-----|
|2012-09-12|SPY   |39.92| 35| 40.0|
|2012-09-11|SPY   |39.60| 45| 39.9|
|2012-09-10|SPY   |38.76| 70| 39.8|

4. Generate Holdings based on Strategy

|Date      |SPY   |AAPL  | IBM  |CASH    |
|----------|------|------|------|--------|
|2012-09-12|    0 |    0 |    0 | 100000 |
|2012-09-11| 1000 |    0 |    0 |  60080 |
|2012-09-10| 1000 | 1000 |-1000 |  70000 |

5. Generate Orders

|Date      |Symbol|Order|Shares|
|----------|------|-----|------|
|2011-01-10|AAPL  |BUY  |1500  |
|2011-01-13|AAPL  |SELL |1500  |
|2011-01-13|IBM   |BUY  |4000  |

6. Backtesting with Orders

|name            |final_strategy_portvals|cum_ret|avg_daily_ret|std_daily_ret|sharpe_ratio|
|----------------|-----------------------|-------|-------------|-------------|------------|
|Strategy Learner|          1.67         | 0.67  | 0.001       | 0.0069      | 2.372      |
|Benchmark       |          1.01         | 0.012 | 0.00017     | 0.017       | 0.157      |
|Manual Strategy |          1.15         | 0.15  | 0.0004      | 0.016       | 0.4        |


## Authors

* **Nan Xiao** - *Initial work* - [trading-book](https://github.com/trading-book)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* Machine Learning for Trading by [Tucker Balch](http://www.cc.gatech.edu/~tucker)
* Machine Learning for Trading [Course](https://quantsoftware.gatech.edu/Machine_Learning_for_Trading_Course)
