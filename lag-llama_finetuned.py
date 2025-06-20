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
plt.savefig("01_temperature_data_overview.png", dpi=300, bbox_inches='tight')
plt.close()

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
    forecast.plot(color='g')
    plt.xticks(rotation=45)
    ax.xaxis.set_major_formatter(date_formater)
    plt.title(f"Temperature Forecast {idx+1}")
    plt.legend()

plt.gcf().tight_layout()
plt.savefig("02_forecast_predictions_overview.png", dpi=300, bbox_inches='tight')
plt.close()

# Evaluate in smaller batches to avoid memory issues
def evaluate_in_batches(tss, forecasts, batch_size=10):
    print("Evaluating in batches to manage memory...")
    all_metrics = []
    
    for i in range(0, len(tss), batch_size):
        print(f"Processing batch {i//batch_size + 1}/{(len(tss) + batch_size - 1)//batch_size}")
        
        batch_tss = tss[i:i+batch_size]
        batch_forecasts = forecasts[i:i+batch_size]
        
        evaluator = Evaluator()
        agg_metrics, ts_metrics = evaluator(iter(batch_tss), iter(batch_forecasts))
        all_metrics.append(agg_metrics)
        
        # Clear memory
        del batch_tss, batch_forecasts, evaluator
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return all_metrics

# Use batched evaluation
batch_metrics = evaluate_in_batches(tss, forecasts, batch_size=5)

# Aggregate metrics from all batches
def aggregate_batch_metrics(batch_metrics):
    """Aggregate metrics from batches using weighted averages"""
    print("\nAggregating metrics from all batches...")
    
    if not batch_metrics:
        print("No batch metrics to aggregate!")
        return None
    
    # Initialize aggregated metrics
    total_samples = 0
    weighted_rmse = 0
    weighted_mase = 0  
    weighted_mape = 0
    weighted_mae = 0
    weighted_msis = 0
    
    # Print individual batch metrics
    print(f"\nBatch-wise metrics:")
    for i, metrics in enumerate(batch_metrics):
        print(f"Batch {i+1}: RMSE={metrics.get('RMSE', 'N/A'):.4f}, "
              f"MASE={metrics.get('MASE', 'N/A'):.4f}, "
              f"MAPE={metrics.get('MAPE', 'N/A'):.4f}")
    
    # Calculate weighted averages (assuming each batch has equal weight for simplicity)
    # In practice, you'd weight by the number of samples in each batch
    batch_count = len(batch_metrics)
    
    for metrics in batch_metrics:
        # Get metrics with default values if missing
        rmse = metrics.get('RMSE', 0)
        mase = metrics.get('MASE', 0)
        mape = metrics.get('MAPE', 0)
        mae = metrics.get('MAE', 0)
        msis = metrics.get('MSIS', 0)
        
        # Simple average (could be improved with proper weighting)
        weighted_rmse += rmse / batch_count
        weighted_mase += mase / batch_count
        weighted_mape += mape / batch_count
        weighted_mae += mae / batch_count
        weighted_msis += msis / batch_count
    
    aggregated_metrics = {
        'RMSE': weighted_rmse,
        'MASE': weighted_mase,
        'MAPE': weighted_mape,
        'MAE': weighted_mae,
        'MSIS': weighted_msis
    }
    
    return aggregated_metrics

# Aggregate the batch metrics
agg_metrics = aggregate_batch_metrics(batch_metrics)

if agg_metrics:
    print("\n" + "="*50)
    print("FINAL AGGREGATED METRICS:")
    print("="*50)
    
    # Extract specific metrics
    rmse = agg_metrics.get('RMSE', 'N/A')
    mase = agg_metrics.get('MASE', 'N/A')
    mape = agg_metrics.get('MAPE', 'N/A')
    mae = agg_metrics.get('MAE', 'N/A')
    msis = agg_metrics.get('MSIS', 'N/A')
    
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Scaled Error (MASE): {mase:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}%")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Scaled Interval Score (MSIS): {msis:.4f}")
    print("="*50)
    
    # Interpretation guide
    print("\nMETRIC INTERPRETATION:")
    print(f"• RMSE: {rmse:.4f}°C - Average prediction error magnitude")
    print(f"• MASE: {mase:.4f} - {'Good' if mase < 1 else 'Poor'} (< 1 is better than naive forecast)")
    print(f"• MAPE: {mape:.2f}% - Average percentage error")
    
else:
    print("Failed to aggregate batch metrics!")

# Create predictions for specific date range (2023-01-08 to 2023-01-14) - Individual day plots
print("\nCreating individual day plots for 2023-01-08 to 2023-01-14...")

start_date = '2023-01-08'
end_date = '2023-01-14'
date_range = pd.date_range(start=start_date, end=end_date, freq='D')

# Create a 3x3 subplot layout for 7 days (with 2 empty spots)
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
axes = axes.flatten()

for day_idx, current_date in enumerate(date_range):
    ax = axes[day_idx]
    
    # Get actual data for this specific day
    day_start = current_date.strftime('%Y-%m-%d')
    day_end = (current_date + pd.Timedelta(days=1) - pd.Timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')
    
    actual_day_data = data[day_start:day_end]
    
    if len(actual_day_data) > 0:
        # Create training data (use previous week as context)
        context_start = current_date - pd.Timedelta(days=7)
        context_end = current_date - pd.Timedelta(hours=1)
        
        context_data = data[context_start:context_end]
        
        if len(context_data) > 24:  # Ensure we have enough context
            # Create dataset for prediction
            train_day_ds = PandasDataset(
                context_data,
                target="Temperature (C)",
                freq="H"
            )
            
            # Create test dataset for this day
            test_day_ds = PandasDataset(
                actual_day_data,
                target="Temperature (C)",
                freq="H"
            )
            
            try:
                # Make predictions for this day
                forecast_it_day, ts_it_day = make_evaluation_predictions(
                    dataset=test_day_ds,
                    predictor=predictor,
                    num_samples=num_samples
                )
                
                forecasts_day = list(forecast_it_day)
                tss_day = list(ts_it_day)
                
                if forecasts_day and tss_day:
                    forecast_day = forecasts_day[0]
                    ts_day = tss_day[0]
                    
                    # Plot actual temperature for the day
                    ax.plot(actual_day_data.index, actual_day_data.values, 'b-', 
                           label='Actual', linewidth=2, alpha=0.8)
                    
                    # Plot predicted temperature
                    forecast_values = forecast_day.mean
                    forecast_time = actual_day_data.index[:len(forecast_values)]
                    ax.plot(forecast_time, forecast_values, 'r-', 
                           label='Predicted', linewidth=2, alpha=0.8)
                    
                    # Add confidence interval
                    if hasattr(forecast_day, 'quantile'):
                        try:
                            lower_bound = forecast_day.quantile(0.1)
                            upper_bound = forecast_day.quantile(0.9)
                            ax.fill_between(forecast_time, lower_bound, upper_bound, 
                                          alpha=0.2, color='red', label='80% CI' if day_idx == 0 else "")
                        except:
                            pass
                    
                    # Calculate metrics for this day
                    if len(forecast_values) == len(actual_day_data):
                        day_rmse = np.sqrt(np.mean((actual_day_data.values - forecast_values) ** 2))
                        day_mape = np.mean(np.abs((actual_day_data.values - forecast_values) / actual_day_data.values)) * 100
                        
                        # Add text box with metrics
                        textstr = f'RMSE: {day_rmse:.2f}°C\nMAPE: {day_mape:.1f}%'
                        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
                        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                               verticalalignment='top', bbox=props)
                
                else:
                    # Fallback to actual data only
                    ax.plot(actual_day_data.index, actual_day_data.values, 'b-', 
                           label='Actual', linewidth=2)
                    ax.text(0.5, 0.5, 'Prediction\nFailed', transform=ax.transAxes, 
                           ha='center', va='center', fontsize=12, 
                           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
                           
            except Exception as e:
                # Fallback to actual data only
                ax.plot(actual_day_data.index, actual_day_data.values, 'b-', 
                       label='Actual', linewidth=2)
                ax.text(0.5, 0.5, f'Error:\n{str(e)[:20]}...', transform=ax.transAxes, 
                       ha='center', va='center', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        else:
            # Not enough context data
            ax.plot(actual_day_data.index, actual_day_data.values, 'b-', 
                   label='Actual', linewidth=2)
            ax.text(0.5, 0.5, 'Insufficient\nContext Data', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    else:
        # No data for this day
        ax.text(0.5, 0.5, 'No Data\nAvailable', transform=ax.transAxes, 
               ha='center', va='center', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
    
    # Formatting
    ax.set_title(f'{current_date.strftime("%Y-%m-%d")} ({current_date.strftime("%A")})', 
                fontsize=12, fontweight='bold')
    ax.set_xlabel('Hour', fontsize=10)
    ax.set_ylabel('Temperature (°C)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))  # Every 4 hours
    
    # Add legend only to the first subplot
    if day_idx == 0:
        ax.legend(fontsize=9)

# Hide unused subplots
for i in range(len(date_range), len(axes)):
    axes[i].set_visible(False)

plt.suptitle('Daily Temperature Predictions: January 8-14, 2023', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig("03_daily_temperature_predictions_2023-01-08_to_2023-01-14.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"Created individual day plots for {len(date_range)} days")

# Create individual day plots for 6 days from test dataset
print("\nCreating individual day plots for 6 days from test dataset...")

# Get the first 6 days from test dataset
test_start_date = test_data.index[0].date()
test_date_range = pd.date_range(start=test_start_date, periods=6, freq='D')

# Create a 2x3 subplot layout for 6 days
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten()

for day_idx, current_date in enumerate(test_date_range):
    ax = axes[day_idx]
    
    # Get actual data for this specific day from test dataset
    day_start = current_date.strftime('%Y-%m-%d')
    day_end = (current_date + pd.Timedelta(days=1) - pd.Timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')
    
    actual_day_data = test_data["Temperature (C)"][day_start:day_end]
    
    if len(actual_day_data) > 0:
        # Create training data (use previous week from training data as context)
        context_start = current_date - pd.Timedelta(days=7)
        context_end = current_date - pd.Timedelta(hours=1)
        
        # Use training data for context (the historical data)
        context_data = data[context_start:context_end]
        
        if len(context_data) > 24:  # Ensure we have enough context
            # Create dataset for prediction
            train_day_ds = PandasDataset(
                context_data,
                target="Temperature (C)",
                freq="H"
            )
            
            # Create test dataset for this day
            test_day_ds = PandasDataset(
                actual_day_data,
                target="Temperature (C)",
                freq="H"
            )
            
            try:
                # Make predictions for this day
                forecast_it_day, ts_it_day = make_evaluation_predictions(
                    dataset=test_day_ds,
                    predictor=predictor,
                    num_samples=num_samples
                )
                
                forecasts_day = list(forecast_it_day)
                tss_day = list(ts_it_day)
                
                if forecasts_day and tss_day:
                    forecast_day = forecasts_day[0]
                    ts_day = tss_day[0]
                    
                    # Plot actual temperature for the day
                    ax.plot(actual_day_data.index, actual_day_data.values, 'b-', 
                           label='Actual', linewidth=2, alpha=0.8)
                    
                    # Plot predicted temperature
                    forecast_values = forecast_day.mean
                    forecast_time = actual_day_data.index[:len(forecast_values)]
                    ax.plot(forecast_time, forecast_values, 'r-', 
                           label='Predicted', linewidth=2, alpha=0.8)
                    
                    # Add confidence interval
                    if hasattr(forecast_day, 'quantile'):
                        try:
                            lower_bound = forecast_day.quantile(0.1)
                            upper_bound = forecast_day.quantile(0.9)
                            ax.fill_between(forecast_time, lower_bound, upper_bound, 
                                          alpha=0.2, color='red', label='80% CI' if day_idx == 0 else "")
                        except:
                            pass
                    
                    # Calculate metrics for this day
                    if len(forecast_values) == len(actual_day_data):
                        day_rmse = np.sqrt(np.mean((actual_day_data.values - forecast_values) ** 2))
                        day_mape = np.mean(np.abs((actual_day_data.values - forecast_values) / actual_day_data.values)) * 100
                        
                        # Add text box with metrics
                        textstr = f'RMSE: {day_rmse:.2f}°C\nMAPE: {day_mape:.1f}%'
                        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
                        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                               verticalalignment='top', bbox=props)
                
                else:
                    # Fallback to actual data only
                    ax.plot(actual_day_data.index, actual_day_data.values, 'b-', 
                           label='Actual', linewidth=2)
                    ax.text(0.5, 0.5, 'Prediction\nFailed', transform=ax.transAxes, 
                           ha='center', va='center', fontsize=12, 
                           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
                           
            except Exception as e:
                # Fallback to actual data only
                ax.plot(actual_day_data.index, actual_day_data.values, 'b-', 
                       label='Actual', linewidth=2)
                ax.text(0.5, 0.5, f'Error:\n{str(e)[:20]}...', transform=ax.transAxes, 
                       ha='center', va='center', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        else:
            # Not enough context data
            ax.plot(actual_day_data.index, actual_day_data.values, 'b-', 
                   label='Actual', linewidth=2)
            ax.text(0.5, 0.5, 'Insufficient\nContext Data', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    else:
        # No data for this day
        ax.text(0.5, 0.5, 'No Data\nAvailable', transform=ax.transAxes, 
               ha='center', va='center', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
    
    # Formatting
    ax.set_title(f'{current_date.strftime("%Y-%m-%d")} ({current_date.strftime("%A")})', 
                fontsize=12, fontweight='bold')
    ax.set_xlabel('Hour', fontsize=10)
    ax.set_ylabel('Temperature (°C)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))  # Every 4 hours
    
    # Add legend only to the first subplot
    if day_idx == 0:
        ax.legend(fontsize=9)

plt.suptitle('Daily Temperature Predictions: Test Dataset (2025)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig("04_daily_test_dataset_predictions.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"\nCreated individual day plots for {len(test_date_range)} days from test dataset")
print("\nAll plots have been saved as PNG files:")
print("- 01_temperature_data_overview.png")
print("- 02_forecast_predictions_overview.png") 
print("- 03_daily_temperature_predictions_2023-01-08_to_2023-01-14.png")
print("- 04_daily_test_dataset_predictions.png")