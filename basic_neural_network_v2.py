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
def train_regression_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    # Move model to the specified device
    model.to(device)

    # To track the history of training losses
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()  # Make model into training mode
        train_loss = 0

        # YOU HAVE TO USE THE DATALOADER TO PERFORM BATCH TRAINING!!!
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
        # ===============================================
        if epoch % 10 == 0 or epoch == num_epochs-1:
            print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    return model, train_losses, val_losses
def test_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputst = model(inputs)
            losst = criterion(outputst, targets)
            test_loss += losst.item() * inputs.size(0)
    test_loss /= len(test_loader.dataset)
    return test_loss


start_time = time.time()
#===========================================================
dataset =  MergedDataset() #******
# Set the sizes for training, validation, and  testing
train_size = int(0.6 * len(dataset))  # 60% for training
val_size = int(0.2 * len(dataset))    # 20% for validation
test_size = len(dataset) - train_size - val_size  # Remaining 20% for testing

# Split the dataset

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders for each subset
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
#******
#===============================================================
# create an instance for the model
basic_model = Model()
criterion = nn.MSELoss()
# CHOOSE ADAM OPTIMIZER
optimizer = torch.optim.Adam(basic_model.parameters(), lr=0.001)
epochs = 250

trained_model, losses, val_losses = train_regression_model(model=basic_model,
                                               train_loader=train_loader,
                                               val_loader=val_loader,
                                               criterion=criterion,
                                               optimizer=optimizer,
                                               num_epochs=epochs,
                                               device=device)
#===========================================================

#Evaluate on the test set
test_loss = test_model(model=trained_model,
                       test_loader=test_loader,
                       criterion=criterion,
                       device=device)
print(f"mean test loss: {test_loss:.4f}")
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