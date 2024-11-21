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

from torch.utils.data import random_split
#=======================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device is {device}.')
#=======================================================

# This function sets all the required seeds to ensure the experiments are reproducible. Use it in your main code-file.
def set_seed(seed):
    # Set the seed for generating random numbers in Python
    random.seed(seed)
    # Set the seed for generating random numbers in NumPy
    np.random.seed(seed)
    # Set the seed for generating random numbers in PyTorch (CPU)
    torch.manual_seed(seed)
    # If you are using GPUs, set the seed for generating random numbers on all GPUs
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure that all operations on GPU are deterministic (if possible)
    torch.backends.cudnn.deterministic = True
    # Disable the inbuilt cudnn auto-tuner that finds the best algorithm to use for your hardware
    torch.backends.cudnn.benchmark = False


seed_num = 41
set_seed(seed_num)
#===============================================================

# Create a model class that inherits nn.Module -- NEURAL NETWORK
class Model(nn.Module):
    # input layer ( 2 features, Power Pressure) --> Hidden Layer 1 (H1) --> H2 --> Output (25 outputs)
    def __init__(self, in_features=2, h1=10, h2=10, out_features=25):
        super().__init__()  # instantiate our nn.Module, always have to do it
        self.fc1 = nn.Linear(in_features, h1) # we suppose fully connected layer (fc-> fully connected)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    # we need now the function to move everything forward
    def forward(self, x):
        # we choose relu
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
#=========================================================
# Define a generic training loop function
def train_regression_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, patience, scheduler=None):
    # Move model to the specified device
    model.to(device)

    # To track the history of training losses
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    early_counter = 0  # early-stopping counter
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()  # Make model into training mode
        train_loss = 0


        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            train_loss += loss.item() * inputs.size(0) # x batch_size to account for the loss that is the avg per batch

            loss.backward()
            optimizer.step()

        # Calculate and store average loss
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        #Validation step===============================================
        model.eval() #turn into evaluation mode
        val_loss = 0

        with torch.no_grad(): #disable gradient calc
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputsv = model(inputs)        #v from validation
                lossv = criterion(outputsv, targets)
                val_loss += lossv.item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        # Adjust learning rate with ReduceLROnPlateau
        if scheduler:
            scheduler.step(val_loss)

        # ===============================================
        if epoch % 10 == 0 or epoch == num_epochs-1:
            print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        #Early stopping implementation
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_counter = 0 #reset the counter if validation loss improves
            best_model_state = model.state_dict()
        else:
            early_counter += 1
            if early_counter >= patience:
                print(f"Early stopping triggered at {epoch + 1} epoch")
                model.load_state_dict(best_model_state)
                break

    return model, train_losses, val_losses

def test_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    all_predictions, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputst = model(inputs)
            losst = criterion(outputst, targets)
            test_loss += losst.item() * inputs.size(0)
            all_predictions.append(outputst.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    test_loss /= len(test_loader.dataset)
    return test_loss, all_predictions, all_targets


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

epochs = 250

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

#test results
 #Convert 2D arrays into a DataFrame with separate columns for each feature
def unscale(data, column_names, scaling_info):
    unscaled_data = data.copy()
    num_columns = data.shape[1]
    for i, column in enumerate(column_names):
        if i >= num_columns:  # Check if index exceeds the number of columns
            print(f"Warning: Index {i} is out of bounds for data with {num_columns} columns.")
            break
        mean = scaling_info[column]['mean']
        std = scaling_info[column]['std']
        unscaled_data[:, i] = (data[:, i] * std) + mean  # Reverse normalization
    return unscaled_data

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
plt.plot(list(range(epochs)), losses, label="Training Loss")
plt.plot(list(range(epochs)), val_losses, label="Validation Loss")
plt.ylabel("Total Cost")
plt.xlabel("Epoch")
plt.title("Training & Validation Loss Progression")
plt.legend()
plt.rcParams.update({'font.size': 22})
plt.show()

#save the model to dictionary
model_save_path = 'trained_model1.pth'
torch.save(trained_model.state_dict(), model_save_path)
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