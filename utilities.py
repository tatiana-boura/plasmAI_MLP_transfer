import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Device setup (CUDA or CPU)
def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device is {device}.')
    return device


# Set seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_regression_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, patience, scheduler):
    # Move model to the specified device
    model.to(device)

    # To track the history of training losses
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    early_counter = 0  # early-stopping counter
    best_model_state = None
    # mae_scores = []
    # mse_scores = []

    for epoch in range(num_epochs):
        model.train()  # Make model into training mode
        train_loss = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            assert outputs.shape == targets.shape, f"Output shape {outputs.shape} doesn't match target shape {targets.shape}"
            loss = criterion(outputs, targets)

            train_loss += loss.item() * inputs.size(0) # x batch_size to account for the loss that is the avg per batch

            loss.backward()
            optimizer.step()

        # Calculate and store average loss
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        #Validation step
        model.eval() #turn into evaluation mode
        val_loss = 0
        r2_values = []

        with torch.no_grad(): #disable gradient calc
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputsv = model(inputs)        #v from validation
                lossv = criterion(outputsv, targets)
                val_loss += lossv.item() * inputs.size(0)

                #r^2 calculation
                for col in range(outputsv.shape[1]):
                    pred_col = outputsv[:, col].cpu().numpy()  # Get predictions for the current column
                    target_col = targets[:, col].cpu().numpy()  # Get targets for the current column

                    # Calculate the residual sum of squares (RSS) and total sum of squares (TSS)
                    rss = ((target_col - pred_col) ** 2).sum()  # Residual Sum of Squares
                    tss = ((target_col - target_col.mean()) ** 2).sum()  # Total Sum of Squares
                    # Calculate R^2 using the formula
                    r2 = 1 - (rss / tss)
                    r2_values.append(r2)



        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        # Adjust learning rate with ReduceLROnPlateau
        if scheduler:
            scheduler.step(val_loss)
            current_lr = scheduler.get_last_lr()[0]  # Get the learning rate of the first group
        else:
            current_lr = optimizer.param_groups[0]['lr']  # Fallback if no scheduler is provided

        if epoch % 10 == 0 or epoch == num_epochs-1:
            print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Learning Rate: {current_lr:.4e}" )
            for col in range(outputs.shape[1]):
                print(f"  R^2 for column {col + 1}: {r2_values[col]:.4f}")
        # Early stopping implementation
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


class Model(nn.Module):
    # input layer ( 2 features, Power Pressure) --> Hidden Layer 1 (H1) --> H2 --> Output (25 outputs)
    def __init__(self, in_features=2, h1=10, h2=10, out_features=10):
        super().__init__()  # instantiate our nn.Module, always have to do it
        self.fc1 = nn.Linear(in_features, h1) # we suppose fully connected layer (fc-> fully connected)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    # we need now the function to move everything forward
    def forward(self, x):
        # we choose relu
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.out(x)
        return x


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


def calculate_weighted_mse(predicted_values, true_values):
    # Ensure that the inputs are numpy arrays
    true_values = true_values.clone().detach() #(true_values, dtype=torch.float32)
    predicted_values = predicted_values.clone().detach().requires_grad_(True)



    # weights_tensor = torch.tensor([
    #     0.093312122, 0.102769082, 0.105229524, 0.118937088, 0.13627972,
    #     0.144720322, 0.153479654, 0.107621811, 0.035236323, 0.002414354
    # ], dtype=torch.float32)

    # Compute the squared differences (error) for each column
    squared_diffs = (true_values - predicted_values) ** 2

    # Apply the weights to each column's squared error
    weighted_squared_diffs = squared_diffs  #* weights_tensor

    # Calculate the mean of the weighted squared differences (weighted MSE)
    weighted_mse = weighted_squared_diffs.mean()

    return weighted_mse


def calculate_huber_loss(predicted_values, true_values, delta=1.0):
    true_values = true_values.clone().detach()  # (true_values, dtype=torch.float32)
    predicted_values = predicted_values.clone().detach().requires_grad_(True)
    diff = torch.abs(predicted_values - true_values)

    # Compute the Huber loss
    loss = torch.where(diff < delta,
                       0.5 * diff ** 2,  # If difference is less than delta, use squared loss
                       delta * (diff - 0.5 * delta))  # Otherwise, use linear loss
    loss[:, :6] *= 4
    return loss.mean()
