import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset_class_V2 import MergedDataset
import json
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utilities import setup_device, test_model, Model, unscale

device = setup_device()
basic_model = Model()
dataset_test = MergedDataset('test_data_no_head_outer_corner_O2.csv')
test_loader = DataLoader(dataset_test, batch_size=16, shuffle=False)
basic_model.load_state_dict(torch.load('trained_model1_O2.pth'))
trained_model = basic_model
criterion = nn.SmoothL1Loss() #calculate_huber_loss #
# Evaluate on the test set
test_loss, all_predictions, all_targets = test_model(model=trained_model,
                                                     test_loader=test_loader,
                                                     criterion=criterion,
                                                     device=device)
print(f"mean test loss: {test_loss:.4f}")
print(type(all_targets))
print(all_predictions.shape)
print(all_targets.shape)

with open('column_stats.json', 'r') as f:
    scaling_info = json.load(f)
output_columns = list(scaling_info.keys())
output_columns = [col for col in output_columns if col not in ['Power', 'Pressure']]
print(type(output_columns))

print(f"all_predictions shape: {all_predictions.shape}")
print(f"Length of output_columns: {len(output_columns)}")

unscaled_predictions = unscale(all_predictions, output_columns, scaling_info)
predictions_df = pd.DataFrame(unscaled_predictions, columns=output_columns)

# Save the DataFrame to CSV file
predictions_df.to_csv('unscaled_predictions.csv', index=False, sep=';')
print("Unscaled predictions have been saved to CSV files.")

# now read the csv's and compute r2 score
# If csv's contain headers, no need for header=None
model_predictions = pd.read_csv('unscaled_predictions.csv', sep=';')
real_targets = pd.read_csv('test_data_no_head_outer_corner_O2.csv',
                           usecols=lambda column: column not in ['Power', 'Pressure'], sep=';')

def calculate_r2(true_values, predicted_values):
    # Ensure true_values and predicted_values are numpy arrays
    true_values = np.array(true_values)
    predicted_values = np.array(predicted_values)

    # Calculate the total sum of squares (TSS)
    mean_true = np.mean(true_values)
    total_sum_of_squares = np.sum((true_values - mean_true) ** 2)

    # Calculate the residual sum of squares (RSS)
    residual_sum_of_squares = np.sum((true_values - predicted_values) ** 2)

    # Calculate R^2
    r2 = 1 - (residual_sum_of_squares / total_sum_of_squares)
    return r2

# calculate R^2 for each column pair
r2_scores = []
for col in range(model_predictions.shape[1]):  # Loop through columns
    pred_col = model_predictions.iloc[:, col].values  # Get predictions for the current column
    target_col = real_targets.iloc[:, col].values  # Get targets for the current column
    # Compute R^2 for the current column pair
    r2 = calculate_r2(target_col, pred_col)
    r2_scores.append(r2)
# Step 4: Display the R^2 scores for each column pair
for i, r2 in enumerate(r2_scores):
    print(f'R^2 for column pair {i + 1}: {r2}')

plt.figure(figsize=(10, 6))  # Set the size of the figure
plt.bar(list(range(1, len(r2_scores) + 1)), r2_scores, color='skyblue', edgecolor='black')
# Add labels and title
plt.xlabel('Column Pair Index', fontsize=22)
plt.ylabel('R^2 Score', fontsize=22)
plt.title('R^2 Scores for Each Column Pair (Prediction vs Target)', fontsize=22)

# Display the plot
plt.xticks(list(range(1, len(r2_scores) + 1)))  # Set x-ticks for each column pair
plt.tight_layout()
plt.show()

# Export the testing data and predictions to CSV with ';' delimiter
test_data = real_targets.copy()  # Copy the actual target values

# Add predictions to the dataframe
predictions_df2 = model_predictions
test_data = test_data.reset_index(drop=True)  # Ensure the indices match
predictions_df2 = predictions_df2.reset_index(drop=True)  # Ensure the indices match

# Concatenate the actual targets and predictions side by side
test_data_with_predictions = pd.concat([test_data, predictions_df2], axis=1)

# Save the concatenated DataFrame to CSV with ';' delimiter
test_data_with_predictions.to_csv('test_data_with_predictions.csv', index=False, sep=';')

print("Testing data with predictions has been saved to 'test_data_with_predictions.csv'.")

# Compute the MSE for each column (i.e., each prediction-target pair)
elementwise_errors = []
mse_scores = []
target_range = []

for col in range(model_predictions.shape[1]):  # Loop through columns
    pred_col = model_predictions.iloc[:, col].values  # Get predictions for the current column
    target_col = real_targets.iloc[:, col].values  # Get targets for the current column
    mse = mean_squared_error(target_col, pred_col)  # Compute MSE for the current column
    mse_scores.append(mse)
    target_range.append((min(target_col), max(target_col)))

    elementwise_errors.append((target_col - pred_col) ** 2)  # Compute the element-wise vectors for mean and std

elementwise_errors = np.array(elementwise_errors)  # Shape: (n_columns, n_samples)

# Display the MSE for each column
for i, mse in enumerate(mse_scores):
    print(f'Column {i+1} | MSE : {mse} and its std {np.std(elementwise_errors[i])}, where values ranged in '
          f'[{target_range[i][0]}, {target_range[i][1]}]')

# Plot MSE Histogram
plt.figure(figsize=(10, 6))  # Set the size of the figure
plt.bar(list(range(1, len(mse_scores) + 1)), mse_scores, color='skyblue', edgecolor='black')

# Add labels and title
plt.xlabel('Column Index', fontsize=22)
plt.ylabel('MSE', fontsize=22)
plt.title('MSE for Each Column Pair (Prediction vs Target)', fontsize=22)
plt.show()

# Compute the MAE for each column (i.e., each prediction-target pair)
mae_scores = []
for col in range(model_predictions.shape[1]):  # Loop through columns
    pred_col = model_predictions.iloc[:, col].values  # Get predictions for the current column
    target_col = real_targets.iloc[:, col].values  # Get targets for the current column
    mae = mean_absolute_error(target_col, pred_col)  # Compute MAE for the current column
    mae_scores.append(mae)

# Display the MAE for each column
for i, mae in enumerate(mae_scores):
    print(f'MAE for column pair {i + 1}: {mae}')

# Plot MAE Histogram
plt.figure(figsize=(10, 6))  # Set the size of the figure
plt.bar(list(range(1, len(mae_scores) + 1)), mae_scores, color='skyblue', edgecolor='black')

# Add labels and title
plt.xlabel('Column Index', fontsize=22)
plt.ylabel('MAE', fontsize=22)
plt.title('MAE for Each Column Pair (Prediction vs Target)', fontsize=22)

# Display the plot
plt.xticks(list(range(1, len(mae_scores) + 1)))  # Set x-ticks for each column pair
plt.tight_layout()
plt.show()

# Plot residuals
residuals = real_targets - model_predictions
residuals_flatten = residuals.values.flatten()

# Plot residuals using a scatter plot (for each data point)
plt.figure(figsize=(10, 6))
plt.scatter(list(range(len(residuals_flatten))), residuals_flatten, color='blue', alpha=0.5)
plt.axhline(y=0, color='black', linestyle='--')  # Horizontal line at 0 for reference
plt.title('Residuals Plot')
plt.xlabel('Index (Data Points)')
plt.ylabel('Residuals (Actual - Predicted)')
plt.show()

# Visualize distribution of residuals
plt.figure(figsize=(10, 6))
# sns.histplot(residuals, kde=True, color='blue')  # KDE to show the distribution
sns.kdeplot(residuals, color='blue', fill=False)
plt.title('Histogram of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()
