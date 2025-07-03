# Times Series Analysis (& Forecasting) of Weather Data


## Overview

In this repository we explore several machine learning methods trained on the BHT Berlin's university compute cluster to predict weather timeseries data.

This repo was initially prepared as a hobby project by chandlerNick to practice web scraping as seen in `src/CollectData.py`. This data was collected from a friend's BME680 sensor that collects weather data in Germany. Then Nick did some basic time series analysis, experimenting with classical ARIMA methods (and sub model thereof) to practice pandas data manipulation and learn a bit more about time series forecasting.

As we (Luisa, Nataliia, and Nick) were assigned to do a machine learning project to learn about using GPU computing infrastructure, the repository changed as we wanted to use methods that would better utilize the available resources. The presentation `misc/presentation.pdf` explains in more detail the process and methods.

---

## Data - Nick

The data we use for this project was collected from the `slackology.net` website. Basically, as mentioned above, Nick's friend has a weather station in Berlin orchestrated by Rasberry Pis using various sensors. In our project we make use of the BME680 sensor temperature readings from 2023 to 2025. More details to the data collection process can be found in `src/CollectData.py`. The data itself is in `data` and the aforementioned website. There are some basic visualizations in the following notebooks as well as in the presentation `misc/presentation`.

---

## Workflow/Infrastructure - Nick, Luisa, Nataliia

The latter part of this project, as mentioned above was concerning the utilization of modern compute architecture (i.e. GPUs) and the workflow used to interact with these resources. Details to this workflow can be found in `DSWorkflow` such as the utilized docker container specification, the kubernetes deployment configuration, and ssh configurations. This allowed us access to GPUs such as Nvidia V100s, A100s, H100s, and more. For our project, we utilized the V100s and A100s to train and evaluate our models. 

---

## Methods - Nick, Luisa, Nataliia

In the following we detail the methods used and where one can find further information pertaining to each. All were tasked with predicting 24 hours in the future using the previous 168 hours (1 week) of average hourly temperature (C). We evaluated on the metrics MAPE, RMSE, and MASE.

[MAPE Article](https://en.wikipedia.org/wiki/Mean_absolute_percentage_error)
[RMSE Article](https://en.wikipedia.org/wiki/Root_mean_square_deviation)
[MASE Article](https://en.wikipedia.org/wiki/Mean_absolute_scaled_error)


### Baselines - Nataliia

In the file `src/`


### LSTM - Nick

In the file `src/LSTM_HPO.ipynb` the final results and code for the LSTM can be seen. We did 10 trials of bayesian search over the hyperparameters using 5-fold time series cross validation on the training data with optuna to select the best hyperparameters. We then retrained the model using all training data before evaluating on the test set, examining metrics, and visualizing several predictions.


### Lag-LLaMA - Nick & Luisa

This method is concerning the paper: [Lag-LLaMA: Open and Efficient LLM Inference with Local Attention](https://arxiv.org/pdf/2310.08278) which is a time series foundation model and therefore allows zero-shot inference as well as fine-tuned inference. We experiment with both in the following sections

#### Zero Shot - Nick

In the file `src/lag-llama.ipynb` the zero-shot inference on the lag-llama model for our data can be seen. Since no fine tuning is required, we simply define the model, load the test data, get the predictions, evaluate the metrics, and visualize a few predictions.

#### Fine Tuned - Luisa

In the file `src/` 







---

## References
- 📄 [Lag-LLaMA: Open and Efficient LLM Inference with Local Attention](https://arxiv.org/pdf/2310.08278) — ArXiv paper  
- 🎥 [Lag-LLaMA Zero-Shot Explained (IBM)](https://youtu.be/MOOPuizuf6o?si=ZopML7Mf2L0NAySF) — YouTube overview (focus: inference on foundation model)
