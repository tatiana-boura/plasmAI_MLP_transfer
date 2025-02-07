import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset_class_V2 import MergedDataset, MergedDatasetTest
#from dataset_class_V3_minmax import MergedDataset #for min max normalization of data
from utilities import setup_device, set_seed, train_regression_model, Model, Model_dynamic, unscale, calculate_huber_loss, calculate_weighted_mse,  ScaledDataset
from torch.utils.data import random_split, Subset
import time
import os
# This function sets all the required seeds to ensure the experiments are reproducible. Use it in your main code-file.
seed_num = 41
set_seed(seed_num)
device = setup_device()
current_directory = os.path.dirname(os.path.realpath(__file__))

csv_file_path_train = os.path.join(current_directory, 'trained_nn_O2_weightedmse_JAN25(v02_physics_based)', 'train_data_no_head_outer_corner_O2.csv')
statsjson = 'column_stats02Ar.json'
model_save_path = 'M21_trained_model1_Ar_with_O2_data.pth'
neural_network = 'trained_model1_Ar_optim_JAN25.pth' #'trained_model1_O2_optim_weightedJAN25.pth' #load the pretrained model
h1_val = 13
layers = 3
batch = 16

#dataset = ScaledDataset('train_data_no_head_outer_corner_O2.csv', is_training=True) #for min max
# Set the sizes for training, validation, and  testing
dataset = MergedDatasetTest(csv_file_path_train, statsjson)
train_size = int(0.8 * len(dataset))  #80% for training
val_size = len(dataset) - train_size    #20% for validation

# Split the dataset not randomly, keep the same sets everytime
train_dataset = Subset(dataset, range(0, train_size))
val_dataset = Subset(dataset, range(train_size, len(dataset)))

# Create DataLoaders for each subset
train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=True)


# create an instance for the model
model = Model_dynamic(h1=h1_val, num_layers=layers)
criterion = calculate_weighted_mse(reduction='mean')
model.load_state_dict(torch.load(neural_network))
optimizer = torch.optim.Adam(model.parameters(), lr=(0.0009518706463849217/20), weight_decay=1.0131506615948022e-05)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1)
#basic_model.load_state_dict(torch.load('trained_model1_O2.pth'))
epochs = 700
start_time = time.time()
trained_model, losses, val_losses, r2_mean = train_regression_model(model=model,
                                                           train_loader=train_loader,
                                                           val_loader=val_loader,
                                                           criterion=criterion,
                                                           optimizer=optimizer,
                                                           num_epochs=epochs,
                                                           device=device,
                                                           patience=20,
                                                           scheduler=scheduler)
end_time = time.time()
# save the model to dictionary
torch.save(trained_model.state_dict(), model_save_path)
print(f"model saved to {model_save_path}")
print(f"time for training: {(end_time-start_time):.4f}")

# Plot Training loss and validation loss at each Epoch
plt.rcParams.update({'font.size': 20})  # Adjust font size as needed
plt.plot(list(range(len(losses))), losses, label="Training Loss")
plt.plot(list(range(len(losses))), val_losses, label="Validation Loss")
plt.ylabel("Total Loss", fontsize=26)
plt.xlabel("Epoch", fontsize=26)
plt.title("Training & Validation Loss Progression", fontsize=28)
plt.legend(fontsize=20)
#plt.xscale('log') #make axis logarithmic
#plt.yscale('log')
plt.rcParams.update({'font.size': 26})
plt.tick_params(axis='both', which='major', labelsize=22)  # Font size for tick labels
plt.show()
