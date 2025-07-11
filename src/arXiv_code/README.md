# arXiv Paper Idea

Here we have code related to an arXiv paper idea we had where we basically document the whole process from data collection to application of time series models. Essentially, this would be a small survey of time series models on a novel dataset collected using the BME680 sensor.

Overleaf project: https://www.overleaf.com/read/jvfszcjqxfjk#be4d9f

## What we need from this code

For each of the python files (except `collect_data.py`), we need the code to simply intake training data and test data as .csv files and output sample predictions over the test set in the form of graphics as well as the three metrics MAPE, RMSE, and MASE.

It should be runnable in the following way:
1. uv run python3 src/arXiv_code/collect_data.py
2. uv run python3 src/arXiv_code/<model-name>.py

Results should then be available somewhere in the repo so we can make the tables and display the necessary example plots.

We need to work on "relevance", maybe.
