import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset_class_V2 import MergedDataset
import json
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from utilities import setup_device, set_seed, train_regression_model, Model, unscale, calculate_huber_loss
from torch.utils.data import random_split

device = setup_device()

# This function sets all the required seeds to ensure the experiments are reproducible. Use it in your main code-file.
seed_num = 41
set_seed(seed_num)

dataset = MergedDataset('train_data_no_head_outer_corner_O2.csv')
# Set the sizes for training, validation, and  testing
train_size = int(0.8 * len(dataset))  # 80% for training
val_size = len(dataset) - train_size    # 20% for validation

# Split the dataset

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders for each subset
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)


# create an instance for the model
basic_model = Model()
criterion = nn.SmoothL1Loss() #calculate_huber_loss #
# CHOOSE ADAM OPTIMIZER with a decay parameter [L2 Regularization method]
optimizer = torch.optim.Adam(basic_model.parameters(), lr=1E-4, weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.01)
#basic_model.load_state_dict(torch.load('trained_model1_O2.pth'))
epochs = 1000

trained_model, losses, val_losses = train_regression_model(model=basic_model,
                                                           train_loader=train_loader,
                                                           val_loader=val_loader,
                                                           criterion=criterion,
                                                           optimizer=optimizer,
                                                           num_epochs=epochs,
                                                           device=device,
                                                           patience=10,
                                                           scheduler=scheduler)

# save the model to dictionary
model_save_path = 'trained_model1_O2.pth'
torch.save(trained_model.state_dict(), model_save_path)
print(f"model saved to {model_save_path}")


# Plot Training loss and validation loss at each Epoch
plt.plot(list(range(len(losses))), losses, label="Training Loss")
plt.plot(list(range(len(losses))), val_losses, label="Validation Loss")
plt.ylabel("Total Loss")
plt.xlabel("Epoch")
plt.title("Training & Validation Loss Progression")
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.rcParams.update({'font.size': 22})
plt.show()
