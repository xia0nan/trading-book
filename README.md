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

## Authors

* **Nan Xiao** - *Initial work* - [trading_book](https://github.com/trading_book)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Machine Learning for Trading by [Tucker Balch](http://www.cc.gatech.edu/~tucker)
* Machine Learning for Trading [Course](https://quantsoftware.gatech.edu/Machine_Learning_for_Trading_Course)
