import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset_class_V2 import MergedDatasetTest
import json
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utilities import setup_device, test_model, Model, unscale, Model_dynamic, calculate_weighted_mse, ScaledDataset, calculate_r2
import os

#specify test data, neural network and folder
current_directory = os.path.dirname(os.path.realpath(__file__))
csv_file_path_test = os.path.join(current_directory, 'test_with_outputs_randomized_not1to10_oxygen', 'test_data_no_head_outer_corner_rand_O2.csv')
neural_network = 'trained_model1_O2_optim_weighted_rand.pth'
h1_val = 11
layers = 2
batch = 64
statsjson = 'column_stats02_rand.json' #also change the stats file at the MergedDatasetTest
csv_file_path_test
name_of_predictions = 'DELunscaled_predictions1_O2_optim_weighted_rand.csv' #csv file for unscaled predictions to be saved

#make the calculations at the test set
device = setup_device()
basic_model = Model_dynamic(h1=h1_val, num_layers=layers)
dataset_test = MergedDatasetTest(csv_file_path_test,statsjson) #!!don't forget to change the stats file at the MergedDatasetTest
test_loader = DataLoader(dataset_test, batch_size=batch, shuffle=False)
basic_model.load_state_dict(torch.load(neural_network)) #trained_model1_O2_weighted_mse.pth change O2 to Ar if you want to choose the other dataset
trained_model = basic_model
criterion = calculate_weighted_mse(reduction='mean') #nn.SmoothL1Loss() #calculate_huber_loss #
# Evaluate on the test set
test_loss, all_predictions, all_targets = test_model(model=trained_model,
                                                     test_loader=test_loader,
                                                     criterion=criterion,
                                                     device=device)
print(f"mean test loss: {test_loss:.4f}")

with open(statsjson, 'r') as f:
    scaling_info = json.load(f)
output_columns_names = list(scaling_info.keys())
output_columns_names = [col for col in output_columns_names if col not in ['Power', 'Pressure']]
#perform unscale of the predictions
unscaled_predictions = unscale(all_predictions, output_columns_names, scaling_info)
predictions_df = pd.DataFrame(unscaled_predictions, columns=output_columns_names)
predictions_df.to_csv(name_of_predictions, index=False, sep=';')

#now both csv's of test data and predictions should be read for r^2 score
model_predictions = pd.read_csv(name_of_predictions, sep=';')
real_targets = pd.read_csv(csv_file_path_test,
                           usecols=lambda column: column not in ['Power', 'Pressure'], sep=';')

# Print model predictions and real targets
# print("Model Predictions:")
# print(model_predictions.head())
# print("Real Targets:")
# print(real_targets.head())
#manual r^2 calculation
r2_scores = []
r2_scores_sklearn = []
for col in range(model_predictions.shape[1]):  # Loop through columns
    pred_col = model_predictions.iloc[:, col]  # Get predictions for the current column
    target_col = real_targets.iloc[:, col]  # Get targets for the current column
    # Compute R^2 for the current column pair
    r2 = calculate_r2(target_col, pred_col)
    r2_scores.append(r2)
    # Compute R^2 using sklearn
    r2_sklearn = r2_score(target_col, pred_col)
    r2_scores_sklearn.append(r2_sklearn)
#Display the R^2 scores for each column pair
for i, (r2_manual, r2_sklearn) in enumerate(zip(r2_scores, r2_scores_sklearn)):
    print(f'R^2 for column pair {i + 1} (Manual): {r2_manual}')
    print(f'R^2 for column pair {i + 1} (sklearn): {r2_sklearn}')

plt.figure(figsize=(10, 6))  # Set the size of the figure
plt.bar(list(range(1, len(r2_scores) + 1)), r2_scores, color='skyblue', edgecolor='black')
# Add labels and title
plt.xlabel('Column Pair Index', fontsize=22)
plt.ylabel('R^2 Score', fontsize=22)
plt.title('R^2 Scores for Each Column Pair (Prediction vs Target)', fontsize=22)
plt.xticks(list(range(1, len(r2_scores) + 1)))  # Set x-ticks for each column pair
plt.tight_layout()
plt.show()

# Visualize distribution of residuals
residuals = abs(real_targets - model_predictions)/real_targets * 100
residuals_flatten = residuals.values.flatten()
plt.rcParams.update({'font.size': 24})  # Adjust font size as needed
plt.figure(figsize=(10, 6))
# sns.histplot(residuals, kde=True, color='blue')  # KDE to show the distribution
sns.kdeplot(residuals, color='blue', fill=False)
plt.title('Histogram of Residuals')
plt.xlabel('(physics_based – NN)/physics_based %', fontsize=28)
plt.ylabel('Frequency', fontsize=28)
# Set axis limits
plt.ylim(0, 0.10)  # Set the y-axis maximum to 0.10
plt.xlim(-1, 15)    # Set the x-axis to extend to 15%
plt.rcParams.update({'font.size': 24})
plt.tick_params(axis='both', which='major', labelsize=26)
plt.show()




