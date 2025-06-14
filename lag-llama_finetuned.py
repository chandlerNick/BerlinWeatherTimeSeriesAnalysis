# huggingface-cli download time-series-foundation-models/Lag-Llama --local-dir lag-llama --include lag-llama.ckpt


from itertools import islice



import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from matplotlib import pyplot as plt
from tqdm.autonotebook import tqdm

import torch

from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.common import ListDataset

from lag_llama.gluon.estimator import LagLlamaEstimator
from torch.serialization import add_safe_globals
from gluonts.torch.modules.loss import NegativeLogLikelihood
from gluonts.torch.distributions.studentT import StudentTOutput
add_safe_globals([NegativeLogLikelihood, StudentTOutput])


# Read in data
years = ["2020", "2021", "2022", "2023", "2024"]
columns = ["ISO Time", "Temperature (C)", "Humidity (%)", "Pressure (kPa)", "Air Quality (Ohms)"]
all_data = pd.read_csv("data/2019_BME680SensorData.csv", usecols=columns)

for year in years:
    data_filename = f"data/{year}_BME680SensorData.csv"
    read_in_data = pd.read_csv(data_filename, usecols=columns)
    all_data = pd.concat([all_data, read_in_data], ignore_index=True)

print("Data Collected")

# Convert to datetime and create uniform hourly intervals
all_data["ISO Time"] = pd.to_datetime(all_data["ISO Time"])
all_data = all_data.set_index("ISO Time")

# Convert numeric columns to float32
numeric_columns = ["Temperature (C)", "Humidity (%)", "Pressure (kPa)", "Air Quality (Ohms)"]
all_data[numeric_columns] = all_data[numeric_columns].astype(np.float32)

# Resample to hourly intervals and calculate mean
hourly_data = all_data.resample('H').mean()
# Forward fill missing values (if any)
hourly_data = hourly_data.ffill().bfill()

data = hourly_data["Temperature (C)"]

# Plot to verify the data
plt.figure(figsize=(14, 5))
data.plot()
plt.title(f"Hourly Average Temperature (C)")
plt.ylabel("Temperature (C)")
plt.show()

# Model setup

prediction_length = 24  # Predict one day ahead
context_length = prediction_length * 7 # One week context 
num_samples = 20
# Set device to GPU if available, otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
batch_size = 64

ckpt_path = "lag-llama/lag-llama.ckpt"
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

# Process test data similarly
test_data = pd.read_csv("data/2025_BME680SensorData.csv", usecols=columns)
test_data["ISO Time"] = pd.to_datetime(test_data["ISO Time"])
test_data = test_data.set_index("ISO Time")

# Convert numeric columns to float32
test_data[numeric_columns] = test_data[numeric_columns].astype(np.float32)

# Resample test data to hourly intervals
test_data = test_data.resample('H').mean()
test_data = test_data.ffill().bfill()

# Convert data to GluonTS format
train_ds = PandasDataset(
    data,
    target="Temperature (C)",
    freq="H"  # Hourly data
)

test_ds = PandasDataset(
    test_data,
    target="Temperature (C)",
    freq="H"  # Hourly data
)

# Set up the estimator
estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

# Create the estimator with fine-tuning configuration
estimator = LagLlamaEstimator(
    ckpt_path=ckpt_path,
    prediction_length=prediction_length,
    context_length=context_length,
    input_size=estimator_args["input_size"],
    n_layer=estimator_args["n_layer"],
    n_embd_per_head=estimator_args["n_embd_per_head"],
    n_head=estimator_args["n_head"],
    time_feat=estimator_args["time_feat"],
    nonnegative_pred_samples=True,
    aug_prob=0.5,
    lr=5e-4,
    batch_size=batch_size,
    num_parallel_samples=num_samples,
    trainer_kwargs={"max_epochs": 50},
    dropout=0.1  # Add some dropout for regularization
)

# Train the model
print("Starting model training...")
predictor = estimator.train(train_ds, cache_data=True, shuffle_buffer_length=1000)
print("Training completed!")

# Make predictions
print("Making predictions...")
forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_ds,
    predictor=predictor,
    num_samples=num_samples
)

forecasts = list(tqdm(forecast_it, total=len(test_ds), desc="Forecasting batches"))
tss = list(tqdm(ts_it, total=len(test_ds), desc="Ground truth"))

# Visualize predictions
plt.figure(figsize=(20, 10))
date_formater = mdates.DateFormatter('%Y-%m-%d %H:%M')
plt.rcParams.update({'font.size': 12})

# Plot the first few predictions
for idx, (forecast, ts) in islice(enumerate(zip(forecasts, tss)), 4):
    ax = plt.subplot(2, 2, idx+1)
    
    plt.plot(ts[-4 * prediction_length:].to_timestamp(), label="Ground Truth")
    forecast.plot(color='g', label="Prediction")
    plt.xticks(rotation=45)
    ax.xaxis.set_major_formatter(date_formater)
    plt.title(f"Temperature Forecast {idx+1}")
    plt.legend()

plt.gcf().tight_layout()
plt.show()

# Evaluate predictions
print("Calculating metrics...")
evaluator = Evaluator()
agg_metrics, ts_metrics = evaluator(iter(tss), iter(forecasts))

print("\nAggregate Metrics:")
print(agg_metrics)