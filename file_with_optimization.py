import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
from dataset_class_V2 import MergedDataset, MergedDatasetTest
import json
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from utilities import setup_device, set_seed, train_regression_model, test_model, Model, unscale
from torch.utils.data import random_split
import optuna
from optimization_utility import objective, callback  # Import the objective function from optimization_file.py
import os
# =======================================================
device = setup_device()
current_directory = os.path.dirname(os.path.realpath(__file__))
# =======================================================

# This function sets all the required seeds to ensure the experiments are reproducible. Use it in your main code-file.
seed_num = 41
set_seed(seed_num)
csv_file_path_train = os.path.join(current_directory, 'test_with_outputs_randomized_not1to10', 'train_data_no_head_outer_corner_rand_O2.csv')
csv_file_path_test = os.path.join(current_directory, 'test_with_outputs_randomized_not1to10', 'test_data_no_head_outer_corner_rand_O2.csv')
dataset = MergedDataset(csv_file_path_train)
dataset_test = MergedDatasetTest(csv_file_path_test)
# Set the sizes for training, validation, and  testing
train_size = int(0.8 * len(dataset))  # 80% for training
val_size = len(dataset) - train_size    # 20% for validation
# test_size = len(dataset) - train_size - val_size  # Remaining 20% for testing

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
best_r2_trial = {'r2': -float('inf'), 'trial_number': None, 'params': None}

def custom_callback(study, trial):
    global best_r2_trial
    callback(study, trial, best_r2_trial)  # Modify callback to accept best_r2_trial

# Create Optuna study to optimize the objective function
start_time = time.time()
study = optuna.create_study(direction='minimize')  # Minimize validation loss
study.optimize(lambda trial: objective(trial, train_dataset, val_dataset, device, num_of_epochs=600), n_trials=60, callbacks=[custom_callback])  # Number of trials to run
best_trial = study.best_trial #best trial with the minimum validation loss
end_time = time.time()
# Get the R2 score of the best trial
trial_r2 = best_trial.user_attrs.get("mean_r2", None)

# Print the best hyperparameters and the best value found
print(f"Best hyperparameters: {study.best_params}")
print(f"Best validation loss: {study.best_value:.4f}")
print(f"R2 score at the best trial:")
print(trial_r2)
#print(f"Trial with highest R2: {best_r2_trial['trial_number']} | R2: {best_r2_trial['r2']:.4f} | Params: {best_r2_trial['params']}")
print(f"Best trial (validation loss): {best_trial.number}")
print(f"time for optuna study: {(end_time-start_time):.4f}")

#print to json file the results
results = {
    "best_hyperparameters": study.best_params,
    "best_validation_loss": study.best_value,
    "best_r2_trial": {
        "trial_number": best_r2_trial['trial_number'],
        "r2": best_r2_trial['r2'],
        "params": best_r2_trial['params']
    },
    "best_trial_number": best_trial.number,
    #"r2_at_best_trial": best_trial_r2,
    "study_duration": end_time - start_time
}

print(json.dumps(results, indent=4))

# Save results to a JSON file
output_file = "optuna_results_weighted_mse_rand_outputs.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=4)

print("The program is paused. Press Enter to continue.")
input()
print("Continuing the program...")
#===================================================================
# Use the best hyperparameters to train the final model
best_lr = study.best_params['lr']
best_batch_size = study.best_params['batch_size']
best_weight_decay = study.best_params['weight_decay']

# Create DataLoaders for each subset
train_loader = DataLoader(train_dataset, batch_size=best_batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=best_batch_size, shuffle=True)
test_loader = DataLoader(dataset_test, batch_size=16, shuffle=False)

# ===============================================================
# create an instance for the model
basic_model = Model()
criterion = nn.MSELoss()
# CHOOSE ADAM OPTIMIZER with a decay parameter [L2 Regularization method]
optimizer = torch.optim.Adam(basic_model.parameters(), lr=best_lr, weight_decay=best_weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1)

epochs = 600

trained_model, losses, val_losses = train_regression_model(model=basic_model,
                                                           train_loader=train_loader,
                                                           val_loader=val_loader,
                                                           criterion=criterion,
                                                           optimizer=optimizer,
                                                           num_epochs=epochs,
                                                           device=device,
                                                           patience=10,
                                                           scheduler=scheduler)

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

# ===========================================================
end_time = time.time()
print("The time of execution of above program is :", (end_time-start_time), "s")
# Plot Training loss and validation loss at each Epoch
plt.plot(list(range(len(losses))), losses, label="Training Loss")
plt.plot(list(range(len(losses))), val_losses, label="Validation Loss")
plt.ylabel("Total Cost")
plt.xlabel("Epoch")
plt.title("Training & Validation Loss Progression")
plt.legend()
plt.rcParams.update({'font.size': 22})
plt.show()

# save the model to dictionary
model_save_path = 'trained_model1.pth'
torch.save(trained_model.state_dict(), model_save_path)
print(f"model saved to {model_save_path}")

# now read the csv's and compute r2 score
# If csv's contain headers, no need for header=None
model_predictions = pd.read_csv('unscaled_predictions.csv', sep=';')
real_targets = pd.read_csv('test_data_no_head_outer_corner.csv',
                           usecols=lambda column: column not in ['Power', 'Pressure'],
                           sep=';')

# calculate R^2 for each column pair
r2_scores = []
for col in range(model_predictions.shape[1]):  # Loop through columns
    pred_col = model_predictions.iloc[:, col].values  # Get predictions for the current column
    target_col = real_targets.iloc[:, col].values  # Get targets for the current column
    # Compute R^2 for the current column pair
    r2 = r2_score(target_col, pred_col)
    r2_scores.append(r2)
# Step 4: Display the R^2 scores for each column pair
for i, r2 in enumerate(r2_scores):
    print(f'R^2 for column pair {i + 1}: {r2}')

plt.figure(figsize=(10, 6))  # Set the size of the figure
plt.bar(range(1, len(r2_scores) + 1), r2_scores, color='skyblue', edgecolor='black')
# Add labels and title
plt.xlabel('Column Pair Index', fontsize=22)
plt.ylabel('R^2 Score', fontsize=22)
plt.title('R^2 Scores for Each Column Pair (Prediction vs Target)', fontsize=22)

# Display the plot
plt.xticks(range(1, len(r2_scores) + 1))  # Set x-ticks for each column pair
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
mse_scores = []
for col in range(model_predictions.shape[1]):  # Loop through columns
    pred_col = model_predictions.iloc[:, col].values  # Get predictions for the current column
    target_col = real_targets.iloc[:, col].values  # Get targets for the current column
    mse = mean_squared_error(target_col, pred_col)  # Compute MSE for the current column
    mse_scores.append(mse)

# Display the MSE for each column
for i, mse in enumerate(mse_scores):
    print(f'MSE for column pair {i + 1}: {mse}')
# Plot MSE Histogram
plt.figure(figsize=(10, 6))  # Set the size of the figure
plt.bar(range(1, len(mse_scores) + 1), mse_scores, color='skyblue', edgecolor='black')

# Add labels and title
plt.xlabel('Column Pair Index', fontsize=22)
plt.ylabel('MSE', fontsize=22)
plt.title('MSE for Each Column Pair (Prediction vs Target)', fontsize=22)

# Display the plot
plt.xticks(range(1, len(mse_scores) + 1))  # Set x-ticks for each column pair
plt.tight_layout()
plt.show()
