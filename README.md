# Times Series Analysis of Weather Data

Here I experiment with some time series analysis methods on weather data from a friend's at home weather station in Berlin.

The sources I reference for the methods and notes are:
- https://machinelearningmastery.com/time-series-forecasting/
- https://machinelearningmastery.com/time-series-forecasting-methods-in-python-cheat-sheet/


## Goal:

Jupyter Notebook of historical data with prediciton methods (up to 2025)

Python file to collect data from website and put into csv files

data directory holding relevant data

Online processing? (eventually -> continually update data directory in batches)


## Notes On Time Series (Generally)

Components of a Time Series (Jason Brownlee):
- Level: Baseline value if it were a straight line.
- Trend: The optional and often linear increasing or decreasing behavior of the series over time.
- Seasonality: The optional repeating patterns or cycles of behavior over time.
- Noise: The optionaly variability in the observations that cannot be explained by the model.

These components may be combined in different ways.

eg. adding:

y = level + trend + seasonality + noise

