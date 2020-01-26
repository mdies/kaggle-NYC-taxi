# Kaggle NYC taxi trip duration

This is my first approach to Kaggle challenge [NYC taxi trip duration](https://www.kaggle.com/c/nyc-taxi-trip-duration). 

Main code is in 'NYC\_taxi\_trip\_duration.ipynb'. However, I run the model in R ('NYC_taxi_trip_duration_xgboost.R') due to what seems to be a bug causing XGBoost killing the kernel (using python 2.17.14 and XGBoost 0.4.0 build string np19py27\_0, both installed with Anaconda in a MacOS X 10.11.6), documented [here](https://stackoverflow.com/questions/51164771/python-xgboost-kernel-died), [here](https://github.com/dmlc/xgboost/issues/1715), and [here](https://datascience.stackexchange.com/questions/33964/python-xgboost-killing-kernel).

UPDATE (Jan 2020): **OBSOLETE CODE. Revision pending! Re-write in Python 3.7 to just get rid of that bug.**
