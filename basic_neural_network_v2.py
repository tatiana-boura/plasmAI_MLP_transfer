import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.ma.core import outer
from torch.utils.data import DataLoader
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from dataset_class_V2 import MergedDataset
import json
import pandas as pd
from sklearn.metrics import r2_score
from utilities import setup_device, set_seed, train_regression_model, test_model, Model, unscale
from torch.utils.data import random_split
#=======================================================
device = setup_device()
print(f'Device is {device}.')
#=======================================================

# This function sets all the required seeds to ensure the experiments are reproducible. Use it in your main code-file.
seed_num = 41
set_seed(seed_num)
#===============================================================
start_time = time.time()
#===========================================================
dataset =  MergedDataset('train_data_no_head_outer_corner.csv')
dataset_test = MergedDataset('test_data_no_head_outer_corner.csv')
# Set the sizes for training, validation, and  testing
train_size = int(0.8 * len(dataset))  # 80% for training
val_size = len(dataset) - train_size    # 20% for validation
#test_size = len(dataset) - train_size - val_size  # Remaining 20% for testing

# Split the dataset

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders for each subset
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset_test, batch_size=16, shuffle=False)
#******
#===============================================================
# create an instance for the model
basic_model = Model()
criterion = nn.MSELoss()
# CHOOSE ADAM OPTIMIZER with a decay parameter [L2 Regularization method]
optimizer = torch.optim.Adam(basic_model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=True)

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
#===========================================================

#Evaluate on the test set
test_loss, all_predictions, all_targets = test_model(model=trained_model,
                       test_loader=test_loader,
                       criterion=criterion,
                       device=device)
print(f"mean test loss: {test_loss:.4f}")
print(type(all_targets))
print(all_predictions.shape)
print(all_targets.shape)

predictions_dfcsv = pd.DataFrame(all_predictions)
predictions_dfcsv.to_csv('predictions_all.csv', index=False, sep=';')

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

#===========================================================
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

#save the model to dictionary
model_save_path = 'trained_model1.pth'
#torch.save(trained_model.state_dict(), model_save_path)
print(f"model saved to {model_save_path}")

# now read the csv's and compute r2 score
# If csv's contain headers, no need for header=None
model_predictions = pd.read_csv('unscaled_predictions.csv', sep=';')
real_targets = pd.read_csv('test_data_no_head_outer_corner.csv', usecols=lambda column: column not in ['Power', 'Pressure'], sep=';')

#calculate R^2 for each column pair
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